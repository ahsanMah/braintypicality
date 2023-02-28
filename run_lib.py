# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import io
import logging
import os
import pdb
import time

import numpy as np
import tensorflow as tf
import torch
from absl import flags
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

import datasets
import likelihood
import losses
import sampling
import sde_lib
import wandb
from datasets import ants_plot_scores, get_channel_selector, plot_slices

# Keep the import below for registering all model definitions
from models import ddpm3d, models_genesis_pp, ncsnpp3d
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, restore_pretrained_weights, save_checkpoint

FLAGS = flags.FLAGS


def inf_iter(data_loader):
    """
    Little Hack to reset iterator
    """

    data_iter = iter(data_loader)
    epoch = 0
    while True:
        try:
            data = next(data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            logging.info("Finished epoch : %d" % (epoch))
            epoch += 1
            data_iter = iter(data_loader)
            data = next(data_iter)
        yield data


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    scheduler = losses.get_scheduler(config, optimizer)

    state = dict(
        optimizer=optimizer, model=score_model, ema=ema, step=0, scheduler=scheduler
    )

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state["step"])

    if initial_step == 0 and config.training.load_pretrain:
        pretrain_dir = os.path.join(config.training.pretrain_dir, "checkpoint.pth")
        state = restore_pretrained_weights(pretrain_dir, state, config.device)

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(
        config, uniform_dequantization=config.data.uniform_dequantization
    )
    train_iter = inf_iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = inf_iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    channel_selector = get_channel_selector(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    masked_marginals = "mask_marginals" in config.data and config.data.mask_marginals

    train_step_fn = losses.get_step_fn(
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
        masked_marginals=masked_marginals,
        scheduler=scheduler,
    )
    eval_step_fn = losses.get_step_fn(
        sde,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
        masked_marginals=masked_marginals,
    )

    diagnsotic_step_fn = losses.get_diagnsotic_fn(
        sde,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
        masked_marginals=masked_marginals,
    )

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.eval.sample_size,
            config.data.num_channels,
            *config.data.image_size,
        )
        print(f"Sampling shape: {sampling_shape}")
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps
        )

    # Trace model for torch
    # score_model = torch.jit.trace(
    #     score_model, (torch.rand(*sampling_shape), torch.rand(config.eval.sample_size))
    # )
    # pdb.set_trace()
    # exit()
    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        if not isinstance(train_ds, torch.utils.data.DataLoader):
            batch = (
                torch.from_numpy(next(train_iter)["image"]._numpy())
                .to(config.device)
                .float()
            )
            batch = batch.permute(0, 3, 1, 2)
        else:
            batch = next(train_iter)["image"].to(config.device).float()
        batch = scaler(batch)
        batch = channel_selector(batch)

        # Execute one training step
        loss = train_step_fn(state, batch).item()
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss))
            writer.add_scalar("training_loss", loss, step)
            wandb.log({"loss": loss}, step=step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:

            # # FIXME: Add check for torch
            # eval_batch =
            #     torch.from_numpy(next(eval_iter)["image"]._numpy())
            #     .to(config.device)
            #     .float()
            # )
            # eval_batch = eval_batch.permute(0, 3, 1, 2)

            # eval_batch = next(eval_iter)["image"].to(config.device).float()
            eval_loss = 0.0  # torch.zeros(config.eval.batch_size, device=config.device)
            sigma_losses = {}
            # sigma_norms = torch.zeros(config.eval.batch_size)

            n_batches = 0
            for eval_batch in eval_ds:
                eval_batch = eval_batch["image"]

                # Drop last batch
                if eval_batch.shape[0] < config.eval.batch_size:
                    continue

                eval_batch = scaler(eval_batch.to(config.device).float())
                eval_batch = channel_selector(eval_batch)
                eval_loss = eval_loss + eval_step_fn(state, eval_batch).item()

                per_sigma_loss = diagnsotic_step_fn(state, eval_batch)
                for sigma, (loss, norms) in per_sigma_loss.items():
                    # print(norms.shape)
                    if sigma not in sigma_losses:
                        sigma_losses[sigma] = (loss, norms)
                    else:
                        l, n = sigma_losses[sigma]
                        l += loss
                        n = torch.cat((n, norms))
                        # print("Catted:", n.shape)
                        sigma_losses[sigma] = (l, n)

                n_batches += 1

            eval_loss /= n_batches

            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
            writer.add_scalar("eval_loss", eval_loss, step)
            wandb.log({"val_loss": eval_loss}, step=step)

            for t, (sigma_loss, sigma_norms) in sigma_losses.items():
                sigma_loss /= n_batches

                logging.info(f"\t\t\t t: {t}, eval_loss:{ sigma_loss:.5f}")
                writer.add_scalar(f"eval_loss/{t}", sigma_loss, step)

                wandb.log({f"val_loss/{t}": sigma_loss}, step=step)
                wandb.log(
                    {f"score_dist/{t}": wandb.Histogram(sigma_norms.numpy())},
                    step=step,
                )

        # Save a checkpoint periodically and generate samples if needed
        if (
            step != 0
            and step % config.training.snapshot_freq == 0
            or step == num_train_steps
        ):
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth"), state
            )

        # Generate and save samples
        if (
            step != 0
            and config.training.snapshot_sampling
            and step % config.training.sampling_freq == 0
        ):
            logging.info("step: %d, generating samples..." % (step))
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            sample, n = sampling_fn(score_model)
            ema.restore(score_model.parameters())
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            tf.io.gfile.makedirs(this_sample_dir)
            sample = np.clip(
                sample.permute(0, 2, 3, 4, 1).cpu().numpy() * 255, 0, 255
            ).astype(np.uint8)
            logging.info("step: %d, done!" % (step))

            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample.np"), "wb"
            ) as fout:
                np.save(fout, sample)

            fname = os.path.join(this_sample_dir, "sample.png")
            # print("Sample shape:", sample.shape)
            # ants_plot_scores(sample, fname)
            try:
                plot_slices(sample, fname)
                wandb.log({"sample": wandb.Image(fname)})
            except:
                logging.warning("Plotting failed!")

                # nrow = int(np.sqrt(sample.shape[0]))
                # sample = np.clip(
                #     sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255
                # ).astype(np.uint8)
                # image_grid = make_grid(sample, nrow, padding=2)
                # with tf.io.gfile.GFile(
                #     os.path.join(this_sample_dir, "sample.png"), "wb"
                # ) as fout:
                #     save_image(image_grid, fout)


def evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    scheduler = losses.get_scheduler(config, optimizer)

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(
        optimizer=optimizer, model=score_model, ema=ema, step=0, scheduler=scheduler
    )

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:

        # Build data pipeline
        eval_ds, _, _ = datasets.get_dataset(
            config,
            uniform_dequantization=config.data.uniform_dequantization,
            evaluation=True,
        )

        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(
            sde,
            train=False,
            optimize_fn=optimize_fn,
            reduce_mean=reduce_mean,
            continuous=continuous,
            likelihood_weighting=likelihood_weighting,
        )

        diagnsotic_step_fn = losses.get_diagnsotic_fn(
            sde,
            reduce_mean=reduce_mean,
            continuous=continuous,
            likelihood_weighting=likelihood_weighting,
        )

    # Create data loaders for likelihood evaluation.
    # Only evaluate on uniformly dequantized data
    if config.eval.ood_eval or config.eval.enable_bpd:
        train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(
            config, uniform_dequantization=True, evaluation=True
        )

        ds_mapper = {"test": eval_ds_bpd}

        if config.eval.ood_eval:
            # Create data loaders for ood evaluation. Only evaluate on uniformly dequantized data
            inlier_ds_bpd, ood_ds_bpd, _ = datasets.get_dataset(
                config, uniform_dequantization=True, evaluation=True, ood_eval=True
            )
            ds_mapper["inlier"] = inlier_ds_bpd
            ds_mapper["ood"] = ood_ds_bpd

        if config.eval.bpd_dataset.lower() == "train":
            ds_bpd = train_ds_bpd
            bpd_num_repeats = 1
        elif config.eval.bpd_dataset.lower() in ds_mapper:
            # Go over the dataset 5 times when computing likelihood on the test dataset
            ds_bpd = ds_mapper[config.eval.bpd_dataset.lower()]
            bpd_num_repeats = 5
        else:
            raise ValueError(
                f"No bpd dataset {config.eval.bpd_dataset} recognized. `ood_eval` is set to  {config.eval.ood_eval}"
            )

    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd:
        likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (
            config.eval.batch_size,
            config.data.num_channels,
            *config.data.image_size,
        )
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps
        )

    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not tf.io.gfile.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}.pth")
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                time.sleep(120)
                state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(score_model.parameters())

        # Compute the loss function on the full evaluation dataset if loss computation is enabled
        if config.eval.enable_loss:
            eval_batch = next(iter(eval_ds))
            all_losses = []
            per_sigma_loss = diagnsotic_step_fn(state, eval_batch)
            for t, sigma_loss in per_sigma_loss.items():
                logging.info(f"\t\t\t t: {t}, eval_loss:{ sigma_loss:.5f}")

            # for i, batch in enumerate(eval_ds):
            #     eval_batch = (
            #         torch.from_numpy(batch["image"]._numpy()).to(config.device).float()
            #     )
            #     eval_batch = eval_batch.permute(0, 3, 1, 2)
            #     eval_batch = scaler(eval_batch)
            #     eval_loss = eval_step(state, eval_batch)
            #     all_losses.append(eval_loss.item())
            #     if (i + 1) % 100 == 0:
            #         logging.info("Finished %dth step loss evaluation" % (i + 1))

            # # Save loss values to disk or Google Cloud Storage
            # all_losses = np.asarray(all_losses)
            # with tf.io.gfile.GFile(
            #     os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb"
            # ) as fout:
            #     io_buffer = io.BytesIO()
            #     np.savez_compressed(
            #         io_buffer, all_losses=all_losses, mean_loss=all_losses.mean()
            #     )
            #     fout.write(io_buffer.getvalue())

        # Compute log-likelihoods (bits/dim) if enabled
        if config.eval.enable_bpd:
            bpds = []
            for repeat in range(bpd_num_repeats):
                bpd_iter = inf_iter(ds_bpd)  # pytype: disable=wrong-arg-types
                for batch_id in range(len(ds_bpd)):
                    batch = next(bpd_iter)
                    eval_batch = (
                        torch.from_numpy(batch["image"]._numpy())
                        .to(config.device)
                        .float()
                    )
                    eval_batch = eval_batch.permute(0, 3, 1, 2)
                    eval_batch = scaler(eval_batch)
                    bpd = likelihood_fn(score_model, eval_batch)[0]
                    bpd = bpd.detach().cpu().numpy().reshape(-1)
                    bpds.extend(bpd)
                    logging.info(
                        "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f"
                        % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds)))
                    )
                    bpd_round_id = batch_id + len(ds_bpd) * repeat
                    # Save bits/dim to disk or Google Cloud Storage
                    with tf.io.gfile.GFile(
                        os.path.join(
                            eval_dir,
                            f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz",
                        ),
                        "wb",
                    ) as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, bpd)
                        fout.write(io_buffer.getvalue())

        # TODO: Compute some evaluation metrics of the images themselves
        #       LPIPS..?
        # Generate samples
        if config.eval.enable_sampling:
            num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
            for r in range(num_sampling_rounds):
                logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
                tf.io.gfile.makedirs(this_sample_dir)
                samples, n = sampling_fn(score_model)
                print(samples.cpu().numpy().min(), samples.cpu().numpy().max())

                # samples = np.clip(
                #     samples.permute(0, 2, 3, 4, 1).cpu().numpy() * 255.0, 0, 255
                # ).astype(np.uint8)

                samples = samples.permute(0, 2, 3, 4, 1).cpu().numpy()

                samples = samples.reshape(
                    (
                        -1,
                        *config.data.image_size,
                        config.data.num_channels,
                    )
                )
                # Write samples to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                    os.path.join(this_sample_dir, f"unscaled-samples_{r}.npz"), "wb"
                ) as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())

                logging.info("sampling -- ckpt: %d, round: %d - completed!" % (ckpt, r))


def compute_scores(config, workdir, score_folder="score"):

    score_dir = os.path.join(workdir, score_folder)
    tf.io.gfile.makedirs(score_dir)

    # Build data pipeline
    inlier_ds, ood_ds, _ = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
        evaluation=True,
        ood_eval=True,
    )

    train_ds, eval_ds, _ = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
        evaluation=True,
    )

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    channel_selector = get_channel_selector(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Initialize model\
    with torch.no_grad():
        score_model = mutils.create_model(config)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(
            score_model.parameters(), decay=config.model.ema_rate
        )
        scheduler = losses.get_scheduler(config, optimizer)
        state = dict(
            optimizer=optimizer, model=score_model, ema=ema, step=0, scheduler=scheduler
        )

        # Loading latest intermediate checkpoint
        ckpt = config.msma.checkpoint
        if ckpt == -1:  # latest-checkpoint
            checkpoint_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
        else:
            checkpoint_dir = os.path.join(
                workdir, "checkpoints", f"checkpoint_{ckpt}.pth"
            )

        state = restore_checkpoint(checkpoint_dir, state, config.device)
        ema = state["ema"]
        ema.copy_to(score_model.parameters())
        score_fn = mutils.get_score_fn(
            sde, score_model, train=False, continuous=config.training.continuous
        )
        ckpt = state["step"]
        logging.info(f"Loaded checkpoint at step {ckpt}")
        predictor_obj = sampling.ReverseDiffusionPredictor(
            sde, score_fn, probability_flow=False
        )

        del state

    # schedule = config.msma.schedule if "schedule" in config.msma else "linear"
    # if schedule == "geometric":
    #     timesteps = torch.exp(
    #         torch.linspace(
    #             np.log(sde.T), np.log(eps), n_timesteps, device=config.device
    #         )
    #     )
    # else:
    #     timesteps = torch.linspace(sde.T, eps, n_timesteps, device=config.device)

    n_timesteps = config.model.num_scales
    eps = config.msma.min_timestep
    end = config.msma.max_timestep

    if isinstance(sde, sde_lib.subVPSDE):
        msma_sigmas = torch.linspace(eps, end, n_timesteps)
        ts = list(map(sde.noise_schedule_inverse, msma_sigmas))
        timesteps = torch.Tensor(ts).to(config.device)
    else:
        logging.warning(
            f"Inverse-schedule function for SDE {config.training.sde} unknown."
        )
        timesteps = torch.linspace(eps, end, n_timesteps, device=config.device)
        # raise NotImplementedError(
        #     f"Inverse-schedule function for SDE {config.training.sde} unknown."
        # )

    # Taken from ODE Sampler
    def denoise_update(x):
        # Reverse diffusion predictor for denoising
        with torch.no_grad():
            eps = 1e-5
            vec_eps = torch.ones(x.shape[0], device=x.device) * eps
            # _, x = predictor_obj.update_fn(x, vec_eps)
            f, G = predictor_obj.rsde.discretize(x, vec_eps)
            x_mean = x - f
            x = x_mean + predictor_obj.sde._unsqueeze(G)
        return x

    def scorer(score_fn, x, return_norm=True, step=-1):

        if step == -1:
            step = n_timesteps // config.msma.n_timesteps

        sz = len(list(range(0, n_timesteps, step)))

        if config.msma.denoise:
            x = denoise_update(x)

        if return_norm:
            scores = np.zeros((sz, x.shape[0]), dtype=np.float32)
        else:
            scores = np.zeros((sz, *x.shape), dtype=np.float32)

        # Background mask
        mask = (inverse_scaler(x) != 0.0).float()

        with torch.no_grad():
            for i, tidx in enumerate(tqdm(range(0, n_timesteps, step))):
                # logging.info(f"sigma {i}")
                t = timesteps[tidx]
                vec_t = torch.ones(x.shape[0], device=config.device) * t
                std = sde.marginal_prob(torch.zeros_like(x), vec_t)[1].cpu().numpy()
                score = score_fn(x, vec_t)

                if config.msma.apply_masks:
                    score = score * mask

                score = score.cpu().numpy()

                if return_norm:
                    score = (
                        np.linalg.norm(
                            score.reshape((score.shape[0], -1)),
                            axis=-1,
                        )
                        * std
                    )
                else:
                    score = score * std[:, None, None, None, None]

                scores[i, ...] = score.copy()
                # del score

        return scores

    def noise_expectation_scorer(score_fn, x, return_norm=True, step=1):
        sz = len(list(range(0, n_timesteps, step)))

        if return_norm:
            scores = np.zeros((sz, x.shape[0]), dtype=np.float32)
        else:
            scores = np.zeros((sz, *x.shape), dtype=np.float32)

        # Background mask
        mask = (inverse_scaler(x) != 0.0).float()
        n_iters = config.msma.expectation_iters

        with torch.no_grad():
            for i, tidx in enumerate(range(0, n_timesteps, step)):
                # logging.info(f"sigma {i}")
                t = timesteps[tidx]
                vec_t = torch.ones(x.shape[0], device=config.device) * t
                mean, std = sde.marginal_prob(x, t)
                score = torch.zeros_like(x)

                for j in range(n_iters):
                    z = torch.randn_like(x)
                    perturbed_x = mean + sde._unsqueeze(std) * z * 0.9
                    score += score_fn(perturbed_x, vec_t)

                score /= n_iters

                if config.msma.apply_masks:
                    # print("MASKING")
                    score = score * mask

                score = score.cpu().numpy()
                std = std.cpu().numpy()

                if return_norm:
                    score = (
                        np.linalg.norm(
                            score.reshape((score.shape[0], -1)),
                            axis=-1,
                        )
                        * std
                    )
                else:
                    score = score * std  # [:, None, None, None, None]

                scores[i, ...] = score.copy()
                # del score

        return scores

    if config.msma.expectation_iters == -1:
        score_runner = scorer
    else:
        score_runner = noise_expectation_scorer

    dataset_dict = {
        "ood": ood_ds,
        "inlier": inlier_ds,
        "eval": eval_ds,
        "train": train_ds,
    }

    for name, ds in dataset_dict.items():

        if config.msma.skip_inliers and name != "ood":
            continue

        if name == "ood" and "LESION" in config.data.ood_ds:
            config.data.select_channel = 0
            _selector = get_channel_selector(config)
        else:
            _selector = channel_selector

        logging.info(f"Computing scores for {name} set")

        sample_batch = None
        sample_batch_scores = None
        score_norms = []

        for i, batch in enumerate(tqdm(ds)):
            if not isinstance(ds, torch.utils.data.DataLoader):
                x_batch = (
                    torch.from_numpy(batch["image"]._numpy()).to(config.device).float()
                )
                x_batch = x_batch.permute(0, 3, 1, 2)
            else:
                x_batch = batch["image"].to(config.device).float()

            x_batch = scaler(x_batch)
            x_batch = _selector(x_batch)

            if sample_batch is None:
                logging.info(f"Recording first batch for {name} set")
                sample_batch = batch["image"][:8].numpy()
                sample_batch_scores = score_runner(
                    score_fn,
                    x_batch[:8],
                    return_norm=False,
                    step=n_timesteps // 5,  # 5 sigmas used
                )
                logging.info(f"Recording first batch for {name} set - Done!")

            x_score_norms = score_runner(score_fn, x_batch, return_norm=True)
            score_norms.append(x_score_norms)

            # [
            #     :: n_timesteps // 10
            # ]
            # x_score[
            #     :: n_timesteps // 10
            # ]  # .cpu().numpy()[::100]

            if (i + 1) % 10 == 0:
                logging.info("Finished step %d for score evaluation" % (i + 1))

        score_norms = np.concatenate(score_norms, axis=1)

        if name == "ood":
            if config.data.gen_ood:
                name = "gen-ood-small"
            else:
                name = config.data.ood_ds

            # if "ez" in config.data.ood_ds:
            #     name += "-ez"
        elif name in ["inlier", "ood"] and config.data.ood_ds in ["IBIS", "DS-SA"]:
            name = f"{config.data.ood_ds}-{name}"

        # num_timesteps = score_norms.shape[0]
        fname = f"ckpt_{ckpt}_{name}_n{config.msma.n_timesteps}_score_dict.npz"
        if config.msma.expectation_iters > -1:
            fname = f"exp-{config.msma.expectation_iters}_" + fname

        if config.msma.apply_masks:
            fname = "masked-" + fname

        if config.msma.denoise:
            fname = "denoised-" + fname

        with tf.io.gfile.GFile(
            os.path.join(score_dir, fname),
            "wb",
        ) as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(
                io_buffer,
                **{
                    "sample_batch": sample_batch,
                    "sample_scores": sample_batch_scores,
                    "score_norms": score_norms,
                },
            )
            fout.write(io_buffer.getvalue())
