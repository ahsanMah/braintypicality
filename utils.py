import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from numba import jit

from models.ema import ExponentialMovingAverage


def restore_checkpoint(ckpt_dir, state, device):
    if not tf.io.gfile.exists(ckpt_dir):
        tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input"
        )
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        if state.get("optimizer") is not None:
            state["optimizer"].load_state_dict(loaded_state["optimizer"])
            state["optimizer"].param_groups[0]["capturable"] = True

        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]

        if "scheduler" in loaded_state and state["scheduler"] is not None:
            state["scheduler"].load_state_dict(loaded_state["scheduler"])

        if "grad_scaler" in loaded_state and "grad_scaler" in state:
            state["grad_scaler"].load_state_dict(loaded_state["grad_scaler"])

        if "adaptive_loss_fn" in state:
            state["adaptive_loss_fn"].load_state_dict(loaded_state["adaptive_loss_fn"])
            state["adaptive_loss_opt"].load_state_dict(loaded_state["adaptive_loss_opt"])
            state["adaptive_loss_opt"].param_groups[0]["capturable"] = True
            # print("finished LOADING ADAPTIVE!!")
        logging.info(f"Loaded model state at step {state['step']} from {ckpt_dir}")
        return state


def restore_pretrained_weights(ckpt_dir, state, device):
    assert state["step"] == 0, "Can only load pretrained weights when starting a new run"
    assert tf.io.gfile.exists(
        ckpt_dir
    ), "Pretrain weights directory {ckpt_dir} does not exist"

    loaded_state = torch.load(ckpt_dir, map_location=device)
    # state["model"].load_state_dict(loaded_state["model"], strict=False)
    dummy_ema = ExponentialMovingAverage(state["model"].parameters(), decay=0.999)
    dummy_ema.load_state_dict(loaded_state["ema"])
    dummy_ema.lazy_copy_to(state["model"].parameters())
    logging.info(f"Loaded pretrained EMA weights from {ckpt_dir} at {loaded_state['step']}")

    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }

    if state["scheduler"] is not None:
        saved_state["scheduler"] = state["scheduler"].state_dict()

    if state["grad_scaler"] is not None:
        saved_state["grad_scaler"] = state["grad_scaler"].state_dict()

    if "adaptive_loss_fn" in state:
        saved_state["adaptive_loss_fn"] = state["adaptive_loss_fn"].state_dict()
        saved_state["adaptive_loss_opt"] = state["adaptive_loss_opt"].state_dict()

    torch.save(saved_state, ckpt_dir)
    return


def plot_slices(x):
    if isinstance(x, torch.Tensor):
        x = x.permute(0, 2, 3, 4, 1).detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(3, 3),  # creates grid of axes
        direction="row",
        axes_pad=0.05,  # pad between axes in inch.
        share_all=True,
        cbar_location="right",
        cbar_mode=None,
    )

    s = x.shape[2] // 16

    for i, (ax, cax) in enumerate(zip(grid, grid.cbar_axes)):
        # Iterating over the grid returns the Axes.
        im = x[0, :, (i + 2) * s, :, i % 2]
        cmap = "gray"

        im = ax.imshow(im, cmap="gray", vmax=None)
        fig.colorbar(im, cax=cax)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    return


def get_flow_rundir(config, workdir):
    hparams = f"psz{config.flow.patch_size}"
    hparams += f"-nb{config.flow.num_blocks}-gmm{config.flow.gmm_components}-lr{config.flow.lr}-bs{config.training.batch_size}"
    hparams += f"-np{config.flow.patches_per_train_step}-kimg{config.flow.training_kimg}"

    rundir = os.path.join(workdir, "flow", hparams)

    return rundir


@jit(nopython=True)
def count(x):
    return np.count_nonzero(x, axis=1)


@jit(nopython=True)
def pscore(a, score):
    n = len(a)

    # Prepare broadcasting
    score = np.asarray(score)
    left = count(a < score)
    right = count(a <= score)
    plus1 = left < right
    perct = (left + right + plus1) * (50.0 / n)

    return perct


@jit
def get_local_quantiles(x, kernel_sz=3, quantile=0.9):
    b, h, w, d = x.shape
    percentiles = np.zeros((h, w, d))
    s = kernel_sz // 2
    for i in range(s, h - s):
        for j in range(s, w - s):
            for k in range(s, d - s):
                ij = i - s
                jj = j - s
                kj = k - s

                ik = i + s
                jk = j + s
                kk = k + s
                patch = x[:, ij:ik, jj:jk, kj:kk]
                p = np.quantile(patch, q=quantile)
                percentiles[i, j, k] = p
    return percentiles


@jit(nopython=True)
def get_percentile_tensor(x, reference_scores, kernel_sz=3):
    """
    Computes the percentile of a given tensor x at each position with respect to a reference tensor
    Args:
        x: Tensor of shape (b, h, w, d)
        reference_scores: Tensor of shape (b, h, w, d)
        kernel_sz: int, size of the kernel to compute the percentile in
    Returns:
        percentiles: Tensor of shape (b, h, w, d)
    """

    b, h, w, d = x.shape
    percentiles = np.zeros((b, h, w, d))
    s = kernel_sz // 2
    for i in range(s, h - s):
        for j in range(s, w - s):
            for k in range(s, d - s):
                ij = i - s
                jj = j - s
                kj = k - s

                ik = i + s
                jk = j + s
                kk = k + s

                ref_patch = reference_scores[:, ij:ik, jj:jk, kj:kk]
                score_at_pos = np.expand_dims(x[:, i, j, k], axis=1)
                #                 print(score_at_pos.shape)
                p = pscore(ref_patch.ravel(), score_at_pos)
                #                 print(score_at_pos, p)
                percentiles[:, i, j, k] = p
    return percentiles
