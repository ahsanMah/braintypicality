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
        state["optimizer"].param_groups[0]["capturable"] = True
        state["step"] = loaded_state["step"]
        if state["scheduler"] is not None:
            state["scheduler"].load_state_dict(loaded_state["scheduler"])
        if state["grad_scaler"] is not None:
            state["grad_scaler"].load_state_dict(loaded_state["grad_scaler"])

        logging.info(f"Loaded model state at step {state['step']} from {ckpt_dir}")
        return state


def restore_pretrained_weights(ckpt_dir, state, device):
    assert (
        state["step"] == 0
    ), "Can only load pretrained weights when starting a new run"
    assert tf.io.gfile.exists(
        ckpt_dir
    ), "Pretrain weights directory {ckpt_dir} does not exist"

    loaded_state = torch.load(ckpt_dir, map_location=device)
    state["model"].load_state_dict(loaded_state["model"], strict=False)

    logging.info(
        f"Loaded pretrained weights from {ckpt_dir} at {loaded_state['step']} "
    )

    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
        "scheduler": state["scheduler"].state_dict()
        if state["scheduler"] is not None
        else None,
        "grad_scaler": state["grad_scaler"].state_dict()
        if state["grad_scaler"] is not None
        else None,
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
