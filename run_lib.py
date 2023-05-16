### Training and evaluation for score-based generative models ###

import io
import logging
import os
import time

import numpy as np
import tensorflow as tf
import torch
from absl import flags
from robust_loss_pytorch.adaptive import AdaptiveLossFunction
from torch.utils import tensorboard
from tqdm.auto import tqdm

import datasets
import likelihood
import losses
import sampling
import sde_lib
import wandb
from datasets import get_channel_selector, plot_slices

# Keep the import below for registering all model definitions
from models import ddpm3d, models_genesis_pp, ncsnpp3d
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, restore_pretrained_weights, save_checkpoint
from sampling import ReverseDiffusionPredictor, EulerMaruyamaPredictor

import functools
from sampling import (
    shared_predictor_update_fn,
    shared_corrector_update_fn,
    get_corrector,
    get_predictor,
)

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
    grad_scaler = torch.cuda.amp.GradScaler() if config.training.use_fp16 else None

    state = dict(
        optimizer=optimizer,
        model=score_model,
        ema=ema,
        step=0,
        scheduler=scheduler,
        grad_scaler=grad_scaler,
    )

    if config.optim.adaptive_loss:
        adaptive_loss = AdaptiveLossFunction(
            num_dims=config.data.num_channels * np.prod(config.data.image_size),
            float_dtype=torch.float32,
            device=torch.device(0),
        )

        loss_fn_optimizer = torch.optim.Adam(adaptive_loss.parameters(), lr=1e-5)

        state["adaptive_loss_fn"] = adaptive_loss
        state["adaptive_loss_opt"] = loss_fn_optimizer

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
    # Create data normalizer and its inverse
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
        use_fp16=config.training.use_fp16,
        adaptive_loss=config.optim.adaptive_loss,
    )
    eval_step_fn = losses.get_step_fn(
        sde,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
        masked_marginals=masked_marginals,
        use_fp16=config.training.use_fp16,
        adaptive_loss=False,
    )

    diagnsotic_step_fn = losses.get_diagnsotic_fn(
        sde,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
        masked_marginals=masked_marginals,
        use_fp16=config.training.use_fp16,
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
    logging.info(f"Grad Scaler: {grad_scaler.state_dict()}")
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

        # Execute one training step
        loss = train_step_fn(state, batch)

        if torch.isnan(loss):
            logging.info(
                f"Step:{step} Loss is NaN. Grad Scaler: {grad_scaler.state_dict()}"
            )
            # for name, param in score_model.named_parameters():
            #     print(f"Parameter {name} is {param.data.norm()}")

            #     if param.grad is None:
            #         print(f"\t Gradient is None")
            #     elif torch.isnan(param.grad).any():
            #         print(f"\t Gradient is NaN")
            #     else:
            #         print(f"\t Gradient is {param.grad.mean()}")

            continue
        else:
            loss = loss.item()

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss))
            writer.add_scalar("training_loss", loss, step)
            wandb.log({"loss": loss}, step=step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            if "adaptive_loss_fn" in state:
                loss_fn = state["adaptive_loss_fn"]
                alpha_mean = loss_fn.alpha().mean().item()
                scale_mean = loss_fn.scale().mean().item()
                logging.info(
                    "step: %d, loss_alpha: %.5e, loss_scale %.5e"
                    % (step, loss_fn.alpha().mean(), loss_fn.scale().mean())
                )  # view in tensorboard
                writer.add_scalar("loss_alpha", alpha_mean, step)
                writer.add_scalar("loss_scale", scale_mean, step)
                wandb.log(
                    {"loss_alpha": alpha_mean, "loss_scale": scale_mean}, step=step
                )

            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())

            eval_loss = 0.0  # torch.zeros(config.eval.batch_size, device=config.device)
            sigma_losses = {}
            # sigma_norms = torch.zeros(config.eval.batch_size)

            n_batches = 0
            for eval_batch in eval_ds:
                eval_batch = eval_batch["image"].to(config.device)

                # Drop last batch
                if eval_batch.shape[0] < config.eval.batch_size:
                    eval_loss = eval_loss + eval_step_fn(state, eval_batch).item()
                    continue

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

            ema.restore(score_model.parameters())

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

            # if config.optim.scheduler != "skip":
            #     wandb.log(
            #         {
            #             "lr": state["optimizer"].param_groups[0]["lr"],
            #         },
            #         step=step,
            #     )

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
            # sample = np.clip(
            #     sample.permute(0, 2, 3, 4, 1).cpu().numpy() * 255, 0, 255
            # ).astype(np.uint8)
            sample = sample.permute(0, 2, 3, 4, 1).cpu().numpy()
            # smin = np.min(sample, axis=(1, 2, 3), keepdims=True)
            # smax = np.max(sample, axis=(1, 2, 3), keepdims=True)
            # sample = (sample - smin) / (smax - smin)
            # assert sample.min() == 0.0 and sample.max() == 1.0
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

    # Initialize model
    score_model = mutils.create_model(config, log_grads=False)
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
        train_ds, eval_ds, raw_dataset = datasets.get_dataset(
            config,
            uniform_dequantization=config.data.uniform_dequantization,
            evaluation=True,
        )

        num_eval_samples = len(raw_dataset[1])

        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step_fn = losses.get_step_fn(
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
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1, 1):
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

            all_losses = []
            sigma_losses = {}
            torch.random.manual_seed(42)
            for i, eval_batch in enumerate(tqdm(eval_ds)):
                eval_batch = eval_batch["image"].to(config.device)

                eval_loss = eval_step_fn(state, eval_batch).item()
                per_sigma_loss = diagnsotic_step_fn(state, eval_batch)
                
                for sigma, (loss, norms) in per_sigma_loss.items():
                    # print(norms.shape)
                    if sigma not in sigma_losses:
                        sigma_losses[sigma] = ([loss], [norms])
                    else:
                        sigma_losses[sigma][0].append(loss)
                        sigma_losses[sigma][1].append(norms)
                
                all_losses.append(eval_loss)

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
            all_losses = np.asarray(all_losses)
            with tf.io.gfile.GFile(
                os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb"
            ) as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(
                    io_buffer, all_losses=all_losses, mean_loss=all_losses.mean(),
                    sigma_losses=sigma_losses
                )
                fout.write(io_buffer.getvalue())

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
    channel_selector = get_channel_selector(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Initialize model\
    with torch.no_grad():
        score_model = mutils.create_model(config, log_grads=False)
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
        sampling.ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)

        del state

    n_timesteps = config.model.num_scales
    eps = config.msma.min_timestep
    end = config.msma.max_timestep

    schedule = config.msma.schedule if "schedule" in config.msma else "geometric"
    if schedule == "linear":
        logging.info("Using linearly spaced sigmas.")
        msma_sigmas = torch.linspace(
            max(1e-1, sde.sigma_min),
            sde.sigma_max,
            config.msma.n_timesteps,
            device=config.device,
        )
        timesteps = sde.noise_schedule_inverse(msma_sigmas).to(config.device)
    elif schedule == "geometric":
        logging.info("Using geometrically spaced sigmas.")
        msma_sigmas = torch.exp(
            torch.linspace(
                np.log(max(1e-1, sde.sigma_min)),
                np.log(sde.sigma_max),
                config.msma.n_timesteps,
                device=config.device,
            )
        )
        timesteps = sde.noise_schedule_inverse(msma_sigmas).to(config.device)
    else:
        logging.warning("Using timesteps to determine sigmas.")
        timesteps = torch.linspace(eps, end, n_timesteps, device=config.device)
        timesteps = timesteps[:: n_timesteps // config.msma.n_timesteps]
        # raise NotImplementedError(
        #     f"Inverse-schedule function for SDE {config.training.sde} unknown."
        # )

    # @torch.inference_mode()
    # def denoise_update(x):
    #     # Reverse diffusion predictor for denoising
    #     eps = 1e-3
    #     vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    #     # _, x = predictor_obj.update_fn(x, vec_eps)
    #     f, G = predictor_obj.rsde.discretize(x, vec_eps)
    #     print("DENOISE:", f.mean(), G.mean())
    #     x_mean = x - f
    #     # x = x_mean + predictor_obj.sde._unsqueeze(G)
    #     return x

    # @torch.inference_mode()
    # def denoise_update(x):
    #     # LANGEVIN CORRECTOR STEP for denoising
    #     eval_score_fn = mutils.get_score_fn(
    #         sde, score_model, train=False, continuous=config.training.continuous
    #     )
    #     eps = 1e-3
    #     vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    #     grad = eval_score_fn(x, vec_eps).cpu()
    #     # noise = torch.randn_like(x)
    #     grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1)
    #     step_size = (2 / grad_norm)

    # #     print(grad.device, x.device)
    #     x_mean = x.cpu() + sde._unsqueeze(step_size) * grad
    # #     x_mean = x.cpu() + grad.cpu()
    #     # x = x_mean + sde._unsqueeze(torch.sqrt(step_size * 2)) * noise

    #     print(
    #         "DENOISE:",grad_norm.mean(), step_size.mean(),"std:",sde.marginal_prob(x,vec_eps.cpu())[1][0]
    #     )

    #     return x_mean

    PC_DENOISER = True
    DENOISE_STEPS = 10
    DENOISE_EPS = 1e-2

    @torch.inference_mode()
    def denoise_update(x, eps=1e-2):
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * DENOISE_EPS
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    snr = 0.175 / 8
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    pc_denoiser = get_pc_denoiser(
        sde,
        predictor,
        corrector,
        snr=snr,
        n_steps=DENOISE_STEPS,
        continuous=config.training.continuous,
    )

    @torch.no_grad()
    def hscore_fd(score_net, x, eps=0.1):
        """
        calculate hscore by finite difference (Gaussian randoms)
        input:
            score_net: \partial_x logp(x) in shape (n, d1, d2, d3), n_particles << d1*d2*d3
            x: sample mini-batches in shape (n, d1, d2, d3)
            eps: magnitude for perturbation
        output:
            hscore in shape (n)
        """
        dim = x.reshape(x.shape[0], -1).shape[-1]
        xlogp_ = score_net(x)
        vectors = torch.randn_like(x)
        # Scale the variance of vector to be (eps**2*I_{dxd})/dim
        vectors = (
            vectors
            / torch.sqrt(
                torch.sum(vectors**2, dim=tuple(range(1, len(x.shape))), keepdim=True)
            )
            * eps
        )
        out1 = score_net(x + vectors)
        out2 = score_net(x - vectors)
        grad2 = out1 - out2

        # loss_1 = torch.sum((grad1 * grad1) / 8.0, dim=tuple(range(1, len(x.shape))))
        loss_1 = (
            torch.sum(xlogp_ * xlogp_, dim=tuple(range(1, len(x.shape)))) / 2.0
        ).detach()
        loss_2 = (
            torch.sum(
                grad2 * vectors * (dim / (2 * eps * eps)),
                dim=tuple(range(1, len(x.shape))),
            )
        ).detach()
        hscore = loss_1 + loss_2
        return hscore

    @torch.inference_mode()
    def hyv_scorer(score_fn, x, eps=1e-2, return_norm=True):
        batch_sz = x.shape[0]
        scores = torch.zeros(config.msma.n_timesteps, batch_sz)
        vec_t = torch.ones(batch_sz).cuda()

        for i, t in enumerate(timesteps):
            s = hscore_fd(lambda xin: score_fn(xin, vec_t * t), x, eps=eps)
            # s *= msma_sigmas[i]
            scores[i, :] = s.cpu()

        return scores.numpy()

    @torch.inference_mode()
    def scorer(score_fn, x, return_norm=True, step=1):
        """Compute scores for a batch of samples.
        Indexing into the timesteps list grabs the *exact* sigmas used during training
        The alternate would be to compute a linearly spaced list of sigmas of size msma.n_timesteps
        However, this would technicaly output sigmas never seen during training...
        """
        # if step == -1:
        #     step = n_timesteps // config.msma.n_timesteps

        sz = len(list(range(0, config.msma.n_timesteps, step)))
        # sz = config.msma.n_timesteps

        with torch.no_grad():
            if config.msma.denoise is True:
                if PC_DENOISER:
                    x = pc_denoiser(score_model, x)
                else:
                    for i in range(DENOISE_STEPS):
                        x = denoise_update(x)

            if return_norm:
                scores = np.zeros((sz, x.shape[0]), dtype=np.float32)
            else:
                scores = np.zeros((sz, *x.shape), dtype=np.float32)

            # Background mask
            mask = (inverse_scaler(x) != 0.0).float()

            for i, tidx in enumerate(tqdm(range(0, config.msma.n_timesteps, step))):
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

    if config.msma.score_fn == "hyv":
        score_runner = hyv_scorer
    elif config.msma.expectation_iters == -1:
        score_runner = scorer
    else:
        score_runner = noise_expectation_scorer

    dataset_dict = {
        "ood": ood_ds,
        "inlier": inlier_ds,
        "eval": eval_ds,
        # "train": train_ds,
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

            # x_batch = scaler(x_batch)
            # x_batch = _selector(x_batch)

            if sample_batch is None:
                logging.info(f"Recording first batch for {name} set")
                sample_batch = batch["image"][:8].numpy()
                sample_batch_scores = scorer(
                    score_fn,
                    x_batch[:8],
                    return_norm=False,
                    step=config.msma.n_timesteps // 5,  # 5 sigmas used
                )
                logging.info(f"Recording first batch for {name} set - Done!")

            x_score_norms = score_runner(score_fn, x_batch, return_norm=True)
            score_norms.append(x_score_norms)

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
            fname = (
                f"denoised{'-pc' if PC_DENOISER else ''}-{DENOISE_STEPS}@{DENOISE_EPS:.0e}-"
                + fname
            )

        if config.msma.schedule != "skip":
            fname = f"{config.msma.schedule}-" + fname

        if config.msma.score_fn == "hyv":
            fname = "hyv-" + fname

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


def get_pc_denoiser(
    sde,
    predictor,
    corrector,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=True,
    eps=1e-3,
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=1,
    )

    def pc_denoiser(model, x):
        """Modded PC sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples
        """
        with torch.no_grad():
            # Initial sample
            timesteps = torch.linspace(1e-2, eps, n_steps, device=x.device)

            for t in timesteps:
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                _, x_mean = corrector_update_fn(x, vec_t, model=model)
                x, x_mean = predictor_update_fn(x_mean, vec_t, model=model)

            return x_mean

    return pc_denoiser
