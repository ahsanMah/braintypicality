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

"""All functions related to loss computation and optimization.
"""

import pdb
import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE

from torch import Tensor
from typing import Optional, Tuple

avail_optimizers = {
    "Adam": optim.Adam,
    "Adamax": optim.Adamax,
    "AdamW": optim.AdamW,
    "RAdam": optim.RAdam,
}


def get_optimizer(config, params):
    """Returns an optimizer object based on `config`."""
    if config.optim.optimizer in avail_optimizers:
        opt = avail_optimizers[config.optim.optimizer]

        optimizer = opt(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )

    return optimizer


def get_scheduler(config, optimizer):
    """Returns a scheduler object based on `config`."""

    if config.optim.scheduler == "skip":
        scheduler = None

    if config.optim.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(0.3 * config.training.n_iters),
            gamma=0.3,
            verbose=False,
        )

    if config.optim.scheduler == "cosine":
        # Assumes LR in opt is initial learning rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.n_iters,
            eta_min=1e-6,
        )

    print("Using scheduler:", scheduler)
    return scheduler


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(
        optimizer,
        params,
        step,
        scheduler=None,
        lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip,
        amp_scaler=None,
    ):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if step <= warmup:
            for g in optimizer.param_groups:
                g["lr"] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            if amp_scaler is not None:
                amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        if amp_scaler is not None:
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            optimizer.step()

        if step > warmup and scheduler is not None:
            scheduler.step()

    return optimize_fn


def get_generalized_noise(
    x: Tensor, beta: float, sigmas: Optional[Tensor] = torch.tensor(1.0)
) -> Tuple[Tensor, Tensor]:
    if beta == 2.0:  # Corresponds to Standard Normal
        noise = sigmas * torch.randn_like(x, device=sigmas.device)
        score = -1 / (sigmas**2) * noise
    else:
        alpha = 2**0.5
        gamma = np.random.gamma(shape=1 + 1 / beta, scale=2 ** (beta / 2), size=x.shape)
        delta = alpha * gamma ** (1 / beta) / (2**0.5)
        gn_samples = (2 * np.random.rand(*x.shape) - 1) * delta

        noise = sigmas * torch.tensor(gn_samples).float().to(x.device)
        constant = -beta / (sigmas * 2.0**0.5) ** beta
        score = constant * torch.sign(noise) * torch.abs(noise) ** (beta - 1)

    return noise.to(x.device), score.to(x.device)


def get_sde_loss_fn(
    sde,
    train,
    reduce_mean=True,
    continuous=True,
    likelihood_weighting=True,
    eps=1e-5,
    masked_marginals=False,
    amp=False,
):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(
            sde, model, train=train, continuous=continuous, amp=amp
        )
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps

        # Use conditioning mask
        if masked_marginals:
            mask = batch[:, -1:, :, :]
            batch = batch[:, :-1, :, :]
            z = torch.randn_like(batch)
            mean, std = sde.marginal_prob(batch, t)
            # print(mean.shape, mask.shape)
            perturbed_data = mean + std[:, None, None, None] * z
            perturbed_data = perturbed_data * mask
            perturbed_data = torch.cat((perturbed_data, mask), axis=1)
            # print("mask pert", perturbed_data.shape)
        else:
            z = torch.randn_like(batch)
            mean, std = sde.marginal_prob(batch, t)
            perturbed_data = mean + sde._unsqueeze(std) * z

        score = score_fn(perturbed_data, t)

        if not likelihood_weighting:
            losses = torch.square(score * sde._unsqueeze(std) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / sde._unsqueeze(std))
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels)
        target = -noise / (sigmas**2)[:, None, None, None, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas**2
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = (  # FIXME: change to unsqueeze
            sqrt_alphas_cumprod[labels, None, None, None, None] * batch
            + sqrt_1m_alphas_cumprod[labels, None, None, None, None] * noise
        )
        score = model_fn(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(
    sde,
    train,
    optimize_fn=None,
    reduce_mean=False,
    continuous=True,
    likelihood_weighting=False,
    masked_marginals=False,
    scheduler=None,
    use_fp16=False,
):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """

    if continuous:
        loss_fn = get_sde_loss_fn(
            sde,
            train,
            reduce_mean=reduce_mean,
            continuous=True,
            likelihood_weighting=likelihood_weighting,
            masked_marginals=masked_marginals,
            amp=use_fp16,
        )
    else:
        assert (
            not likelihood_weighting
        ), "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(
                f"Discrete training for {sde.__class__.__name__} is not recommended."
            )

    if use_fp16:
        print(f"Using AMP for {'training' if train else 'evaluation'}.")

        def step_fn(state, batch):
            """Running one step of training or evaluation with AMP"""
            model = state["model"]
            if train:
                optimizer = state["optimizer"]
                loss_scaler = state["grad_scaler"]
                optimizer.zero_grad(set_to_none=True)

                loss = loss_fn(model, batch)

                loss_scaler.scale(loss).backward()

                # pdb.set_trace()

                optimize_fn(
                    optimizer,
                    model.parameters(),
                    step=state["step"],
                    scheduler=scheduler,
                    amp_scaler=loss_scaler,
                )
                state["step"] += 1
                state["ema"].update(model.parameters())
            else:
                # Assume that the model is already in eval mode.
                # And that the EMA is already applied.
                with torch.no_grad():
                    loss = loss_fn(model, batch)

            return loss

    else:

        def step_fn(state, batch):
            """Running one step of training or evaluation.

            Args:
            state: A dictionary of training information, containing the score model, optimizer,
            EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data.

            Returns:
            loss: The average loss value of this state.
            """
            model = state["model"]
            if train:
                optimizer = state["optimizer"]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, batch)
                loss.backward()
                optimize_fn(
                    optimizer,
                    model.parameters(),
                    step=state["step"],
                    scheduler=scheduler,
                )
                state["step"] += 1
                state["ema"].update(model.parameters())
            else:
                with torch.no_grad():
                    ema = state["ema"]
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    loss = loss_fn(model, batch)
                    ema.restore(model.parameters())

            return loss

    return step_fn


def get_scorer(sde, continuous=True, eps=1e-5):
    def scorer(model, batch):
        """Compute the weighted scores function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          score: A tensor that represents the weighted scores for each sample in the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        score = score_fn(batch, t)
        return score

    return scorer


def score_step_fn(sde, continuous=True, eps=1e-5):
    scorer = get_scorer(
        sde,
        continuous=continuous,
    )

    def step_fn(state, batch):
        """Running one step of scoring

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
            EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        # FIXME!!!! I was restoring original params back
        model = state["model"]
        ema = state["ema"]
        ema.copy_to(model.parameters())
        with torch.no_grad():
            score = scorer(model, batch)
        return score

    return step_fn


def get_diagnsotic_fn(
    sde,
    reduce_mean=False,
    continuous=True,
    likelihood_weighting=False,
    masked_marginals=False,
    eps=1e-5,
    steps=5,
    use_fp16=False,
):
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def sde_loss_fn(model, batch, t):
        """Compute the per-sigma loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous, amp=use_fp16)
        _t = torch.ones(batch.shape[0], device=batch.device) * t * (sde.T - eps) + eps

        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, _t)
        perturbed_data = mean + sde._unsqueeze(std) * z

        score = score_fn(perturbed_data, _t)
        score_norms = torch.linalg.norm(score.reshape((score.shape[0], -1)), dim=-1)
        score_norms = score_norms * std

        if not likelihood_weighting:
            losses = torch.square(score * sde._unsqueeze(std) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), _t)[1] ** 2
            losses = torch.square(score + z / sde._unsqueeze(std))
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)

        return loss, score_norms

    def smld_loss_fn(model, batch, t):
        model_fn = mutils.get_model_fn(model, train=False)
        labels = torch.ones(batch.shape[0], device=batch.device) * t
        sigmas = smld_sigma_array.to(batch.device)[labels.long()]
        # print(labels.long()[0], sigmas[0])
        noise = torch.randn_like(batch) * sigmas[:, None, None, None, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels)
        score_norms = torch.linalg.norm(score.reshape((score.shape[0], -1)), dim=-1)
        score_norms = score_norms * sigmas

        target = -noise / (sigmas**2)[:, None, None, None, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas**2
        loss = torch.mean(losses)
        return loss, score_norms

    if not continuous:
        assert isinstance(sde, VESDE), "SMLD training only works for VESDEs."
        smld_sigma_array = torch.flip(sde.discrete_sigmas, dims=(0,))
        final_timepoint = sde.N - 1
        loss_fn = smld_loss_fn
    else:
        final_timepoint = 1.0
        loss_fn = sde_loss_fn

    def step_fn(state, batch):
        model = state["model"]
        with torch.no_grad():
            losses = {}

            for t in torch.linspace(0.0, final_timepoint, steps, dtype=torch.float32):
                loss, norms = loss_fn(model, batch, t)
                losses[f"{t:.3f}"] = (loss.item(), norms.cpu())

        return losses

    return step_fn
