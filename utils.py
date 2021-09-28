import torch
import tensorflow as tf
import os
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def restore_checkpoint(ckpt_dir, state, device):
    if not tf.io.gfile.exists(ckpt_dir):
        tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input"
        )
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
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
