import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import functools
import re
import glob
import ants
import antspynet
import pandas as pd
import numpy as np
from time import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from generate_mri import register_and_match, get_hcpdpaths


def seg_runner(path, dataset="ABCD"):
    import tensorflow as tf

    cache_dir = "./template_cache/"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if dataset == "ABCD":
        R = re.compile(r"Data\/sub-(.*)\/ses-")
        subject_id = R.search(path).group(1)
        t1_path = path
        t2_path = path.replace("T1w", "T2w")
    elif dataset == "HCPD":
        subject_id, t1_path = path
        t2_path = t1_path.replace("T1w_", "T2w_")
    else:
        raise NotImplementedError

    t1_img = ants.image_read(t1_path)
    t2_img = ants.image_read(t2_path)
    
    t1_seg = antspynet.utilities.deep_atropos(
        t1_img, antsxnet_cache_directory=cache_dir
    )["segmentation_image"]

    # Rigid regsiter to MNI + hist normalization + min/max scaling
    t1_img, t1_mask, registration = register_and_match(
        t1_img,
        modality="t1",
        antsxnet_cache_directory=cache_dir,
        verbose=False,
    )

    # Register t2 to the t1 already registered to MNI above
    t2_img, t2_mask, _ = register_and_match(
        t2_img,
        modality="t2",
        target_img=t1_img,
        target_img_mask=t1_mask,
        antsxnet_cache_directory=cache_dir,
        verbose=False,
    )

    # Also register segmentations to new t1
    t1_seg = ants.apply_transforms(
        fixed=t1_img,
        moving=t1_seg,
        transformlist=registration["fwdtransforms"],
        interpolator="genericLabel",
    )

    # Further apply the opposite modality masks to get a tighter brain crop
    # the same as t1_mask & t2_mask
    combined_mask = t1_mask * t2_mask
    t1_img = t1_img * combined_mask
    t2_img = t2_img * combined_mask

    wm_mask = t1_seg == 3
    t1_wm = t1_img * wm_mask
    t1_wm = t1_wm[t1_wm > 0].ravel()

    t2_wm = t2_img * wm_mask
    t2_wm = t2_wm[t2_wm > 0].ravel()

    # Save outputs
    fname = os.path.join("/DATA/Users/amahmood/braintyp/segs/", f"{subject_id}.npz")
    np.savez_compressed(fname, **{"t1": t1_wm, "t2": t2_wm})

    preproc_img = ants.merge_channels([t1_img, t2_img])
    fname = os.path.join(
        "/DATA/Users/amahmood/braintyp/processed_v2", f"{subject_id}.nii.gz"
    )
    preproc_img.to_filename(fname)

    return


def run(paths, process_fn, chunksize=2):
    start_idx = 0
    start = time()
    progress_bar = tqdm(
        range(0, len(paths), chunksize),
        total=len(paths) // chunksize,
        initial=0,
        desc="# Processed: ?",
    )

    with ProcessPoolExecutor(max_workers=chunksize) as exc:
        for idx in progress_bar:
            idx_ = idx + start_idx
            result = list(exc.map(process_fn, paths[idx_ : idx_ + chunksize]))
            progress_bar.set_description("# Processed: {:d}".format(idx_))

    print("Time Taken: {:.3f}".format(time() - start))


if __name__ == "__main__":

    save_dir = "/DATA/Users/amahmood/braintyp/segs/"
    os.makedirs(save_dir, exist_ok=True)

    assert sys.argv[1] in ["HCPD", "ABCD"], "Dataset name must be defined"

    if sys.argv[1] == "HCPD":
        file_paths = get_hcpdpaths()
        run(file_paths, functools.partial(seg_runner, dataset="HCPD"))
    else: #get abcd paths
        paths = glob.glob("/DATA/ImageData/Data/*/ses-baselineYear1Arm1/anat/*T1w.nii.gz")
        R = re.compile(r"Data\/sub-(.*)\/ses-")
        clean = lambda x: x.strip().replace("_", "")

        with open("abcd_qc_passing_keys.txt", "r") as f:
            abcd_qc_keys = set([clean(x) for x in f.readlines()])

        file_paths = []
        id_checker = lambda x: R.search(x).group(1) in abcd_qc_keys
        file_paths = list(filter(id_checker, paths))

        assert len(file_paths) == len(abcd_qc_keys)

        run(file_paths, seg_runner)
