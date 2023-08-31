import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor
from time import time

import ants
import antspynet
import nibabel as nib
import numpy as np
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = f"{dir_path}/template_cache/"

# For Docker Images
DATA_DIR = "/DATA/"

T1_REF_IMG_PATH = os.path.join(
    CACHE_DIR, "mni_icbm152_09a/mni_icbm152_t1_tal_nlin_sym_09a.nrrd"
)
T2_REF_IMG_PATH = os.path.join(
    CACHE_DIR, "mni_icbm152_09a/mni_icbm152_t2_tal_nlin_sym_09a.nrrd"
)
MASK_REF_IMG_PATH = os.path.join(
    CACHE_DIR, "mni_icbm152_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nrrd"
)

ants_mni = ants.image_read(f"{CACHE_DIR}/croppedMni152.nii.gz")
t1_ref_img = ants.image_read(T1_REF_IMG_PATH)
t2_ref_img = ants.image_read(T2_REF_IMG_PATH)
ref_img_mask = ants.image_read(MASK_REF_IMG_PATH)

# Use ANTs' tighter cropping
diff = np.array(t1_ref_img.shape) - np.array(ants_mni.shape)
crop_idxs_start, crop_idxs_end = 1 + diff // 2, np.array(t1_ref_img.shape) - diff // 2

t1_ref_img = ants.crop_indices(t1_ref_img, crop_idxs_start, crop_idxs_end)
t2_ref_img = ants.crop_indices(t2_ref_img, crop_idxs_start, crop_idxs_end)
ref_img_mask = ants.crop_indices(ref_img_mask, crop_idxs_start, crop_idxs_end)


def extract_brain_mask(
    image, modality="t1", antsxnet_cache_directory=None, verbose=True
):
    from antspynet.utilities import brain_extraction

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    # Truncating intensity as a preprocessing step following the original ants function
    preprocessed_image = ants.image_clone(image)
    truncate_intensity = (0.01, 0.99)
    if truncate_intensity is not None:
        quantiles = (
            image.quantile(truncate_intensity[0]),
            image.quantile(truncate_intensity[1]),
        )
        if verbose == True:
            print(
                "Preprocessing:  truncate intensities ( low =",
                quantiles[0],
                ", high =",
                quantiles[1],
                ").",
            )

        preprocessed_image[image < quantiles[0]] = quantiles[0]
        preprocessed_image[image > quantiles[1]] = quantiles[1]

    # Brain extraction
    mask = None
    probability_mask = brain_extraction(
        preprocessed_image,
        modality=modality,
        antsxnet_cache_directory=antsxnet_cache_directory,
        verbose=verbose,
    )
    mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
    mask = ants.morphology(mask, "close", 6).iMath_fill_holes()

    return mask


def register_and_match(
    image,
    target_img=None,
    target_img_mask=None,
    label=None,
    truncate_intensity=(0.01, 0.99),
    modality="t1",
    template_transform_type="Rigid",
    antsxnet_cache_directory=CACHE_DIR,
    verbose=True,
):

    """
    Basic preprocessing pipeline for T1-weighted brain MRI adapted from AntsPyNet
    https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/preprocess_image.py

    Arguments
    ---------
    image : ANTsImage
        input image

    truncate_intensity : 2-length tuple
        Defines the quantile threshold for truncating the image intensity

    modality : string or None
        Modality that defines registration and brain extraction using antspynet tools.
        One of "t1" or "t2"

    template_transform_type : string
        See details in help for ants.registration.  Typically "Rigid" or
        "Affine".
    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dictionary with preprocessing information ANTs image (i.e., source_image) matched to the
    (reference_image).

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> preprocessed_image = preprocess_brain_image(image, do_brain_extraction=False)
    """
    from antspynet.utilities import brain_extraction

    assert modality in ["t1", "t2"]

    preprocessed_image = ants.image_clone(image)

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    # Truncate intensity
    if truncate_intensity is not None:
        quantiles = (
            image.quantile(truncate_intensity[0]),
            image.quantile(truncate_intensity[1]),
        )
        if verbose == True:
            print(
                "Preprocessing:  truncate intensities ( low =",
                quantiles[0],
                ", high =",
                quantiles[1],
                ").",
            )

        preprocessed_image[image < quantiles[0]] = quantiles[0]
        preprocessed_image[image > quantiles[1]] = quantiles[1]

    # Brain extraction
    mask = None
    probability_mask = brain_extraction(
        preprocessed_image,
        modality=modality,
        antsxnet_cache_directory=antsxnet_cache_directory,
        verbose=verbose,
    )
    mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
    mask = ants.morphology(mask, "close", 6).iMath_fill_holes()

    # Template normalization
    template_image = t1_ref_img if modality == "t1" else t2_ref_img
    template_img_mask = ref_img_mask

    if target_img is None:
        target_img = template_image
        target_img_mask = ref_img_mask
    # else:
    #     # T1 target img is given, we only need to load T2 Template img
    #     template_image = t2_ref_img

    # Similar to ANTsPyNet we compute the registration via masked images
    target_brain_img = target_img * target_img_mask
    preprocessed_brain_image = preprocessed_image * mask
    registration = ants.registration(
        fixed=target_brain_img,
        moving=preprocessed_brain_image,
        type_of_transform=template_transform_type,
        verbose=verbose,
    )

    # Next we apply the transform to the UNMASKED images
    preprocessed_image = ants.apply_transforms(
        fixed=target_img,
        moving=preprocessed_image,
        transformlist=registration["fwdtransforms"],
        interpolator="linear",
        verbose=verbose,
    )
    mask = ants.apply_transforms(
        fixed=preprocessed_image,
        moving=mask,
        transformlist=registration["fwdtransforms"],
        interpolator="genericLabel",
        verbose=verbose,
    )

    if label:
        label_img = ants.apply_transforms(
            fixed=preprocessed_image,
            moving=label,
            transformlist=registration["fwdtransforms"],
            interpolator="genericLabel",
            verbose=verbose,
        )

    # Note that bias correction takes in UNMASKED images
    if verbose == True:
        print("Preprocessing:  brain correction.")
    n4_output = None
    n4_output = ants.n4_bias_field_correction(
        preprocessed_image,
        mask,
        shrink_factor=4,
        return_bias_field=False,
        verbose=verbose,
    )
    preprocessed_image = n4_output

    # Histogram matching with template
    template_brain_image = template_image * template_img_mask
    preprocessed_image = preprocessed_image * mask
    preprocessed_image = ants.utils.histogram_match_image(
        preprocessed_image,
        template_brain_image,
        number_of_histogram_bins=128,
        number_of_match_points=5,  # Could leave to 1 or 2 if u wanna emphasize intensity during training
    )

    # Min max norm
    preprocessed_image = (preprocessed_image - preprocessed_image.min()) / (
        preprocessed_image.max() - preprocessed_image.min()
    )

    if label:
        return preprocessed_image, label_img

    return preprocessed_image, mask, registration


def preprocessor(sample, save_dir=None):
    # print(SAVE_DIR, TUMOR)
    import tensorflow as tf

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

    subject_id, path = sample

    if DATASET == "TUMOR":
        img = nib.load(path)
        t1_img = ants.from_nibabel(img.slicer[..., 2])
        t2_img = ants.from_nibabel(img.slicer[..., 3])
        _mask = t1_img != 0
    else:
        t1_path = path
        if DATASET == "HCPD":
            t2_path = path.replace("T1w_", "T2w_")
        else:
            t2_path = path.replace("T1w", "T2w")

        t1_img = ants.image_read(t1_path)
        t2_img = ants.image_read(t2_path)

        # t1_mask = extract_brain_mask(
        #     t1_img, antsxnet_cache_directory=CACHE_DIR, verbose=False
        # )
        # t2_mask = extract_brain_mask(
        #     t2_img, antsxnet_cache_directory=CACHE_DIR, verbose=False
        # )

        # Rigid regsiter to MNI + hist normalization + min/max scaling
        t1_img_reg, t1_mask, _ = register_and_match(
            t1_img,
            modality="t1",
            antsxnet_cache_directory=CACHE_DIR,
            verbose=False,
        )

        # Register T2 to the post-registered T1
        t2_img_reg, t2_mask, _ = register_and_match(
            t2_img,
            modality="t2",
            target_img=t1_img_reg,
            target_img_mask=t1_mask,
            antsxnet_cache_directory=CACHE_DIR,
            verbose=False,
        )
    preproc_img = ants.merge_channels([t1_img_reg, t2_img_reg])

    fname = os.path.join(SAVE_DIR, f"{subject_id}.nii.gz")
    preproc_img.to_filename(fname)

    return subject_id


def lesion_preprocessor(sample):
    # print(SAVE_DIR, TUMOR)
    import tensorflow as tf

    # print(f"Using contrast {contrast}")
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

    subject_id, path = sample
    t2_path = path
    t2_img = ants.image_read(t2_path)

    label_path = path.replace("T2w", "T2w_lesion_label")
    # print(label_path)
    label_img = ants.image_read(label_path)

    # if high_contrast:
    t2_img[label_img] = t2_img[label_img] * (contrast / 100)

    # Rigid regsiter to MNI + hist normalization + min/max scaling
    # Additionally register label img to mri
    t2_img, label_img, = register_and_match(
        t2_img,
        label=label_img,
        modality="t2",
        antsxnet_cache_directory=CACHE_DIR,
        verbose=False,
    )

    preproc_img = ants.merge_channels((t2_img, label_img))
    fname = os.path.join(SAVE_DIR, f"{subject_id}.nii.gz")
    preproc_img.to_filename(fname)

    return subject_id


def get_matcher(dataset):

    if dataset == "HCPD":
        return re.compile(r"(HCD\d*)_V1_MR")

    if dataset == "IBIS":
        return re.compile(r"stx_(\d*)_VSA*_*")

    if dataset == "BRATS":
        return re.compile(r"(BRATS_\d*).nii.gz")

    # ABCD adult matcher
    if dataset == "ABCD":
        return re.compile(r"sub-(.*)\/ses-")  # NDAR..?

    matcher = r"neo-\d{4}-\d(-\d)?"

    if dataset == "CONTE2":
        return re.compile(matcher)

    return re.compile("(" + matcher + ")")


def get_abcdpaths(split="train"):

    assert split in ["train", "val", "test", "ood"]

    R = get_matcher("ABCD")

    paths = glob.glob(
        DATA_DIR + "ImageData/Data/*/ses-baselineYear1Arm1/anat/*T1w.nii.gz"
    )
    clean = lambda x: x.strip().replace("_", "")

    inlier_paths = []

    with open(f"{split}_keys.txt", "r") as f:
        inlier_keys = set([clean(x) for x in f.readlines()])

    for path in paths:
        match = R.search(path)
        sub_id = match.group(1)

        if sub_id in inlier_keys:
            inlier_paths.append((sub_id, path))

    print("Collected:", len(inlier_paths))
    return inlier_paths


def get_bratspaths(split="train"):

    R = get_matcher("BRATS")

    paths = glob.glob("/DATA/Users/amahmood/tumor/Task01_BrainTumour/imagesTr/*")

    id_paths = []
    for path in paths:
        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))
    return id_paths


def get_ibispaths(split="train"):

    R = get_matcher("IBIS")

    paths = glob.glob(
        "/ASD/Autism/IBIS/Proc_Data/*/VSA*/mri/registered_stx/sMRI/*T1w.nrrd"
    )
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:

        t2_path = path.replace("T1w", "T2w")
        if not os.path.exists(t2_path):
            continue

        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths


def get_lesionpaths():

    R = re.compile(r"sub-(.*)_ses-")
    lesion_dir = "/DATA/Users/amahmood/lesions/lesion_load_50_automated/*T2w.nii.gz"
    paths = sorted(glob.glob(lesion_dir))

    id_paths = []
    for path in paths:
        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths

def get_hcpdpaths(split="train"):

    R = get_matcher("HCPD")
    
    paths = glob.glob(
        "/UTexas/HCP/HCPD/fmriresults01/*_V1_MR/T1w/T1w_acpc_dc.nii.gz"
    )
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:

        t2_path = path.replace("T1w_", "T2w_")
        if not os.path.exists(t2_path):
            continue

        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths


def run(paths, process_fn):
    start = time()
    progress_bar = tqdm(
        range(0, len(paths), chunksize),
        total=len(paths) // chunksize,
        initial=0,
        desc="# Processed: ?",
    )

    with ProcessPoolExecutor(max_workers=cpus) as exc:
        for idx in progress_bar:
            idx_ = idx + start_idx
            # print(paths[idx_ : idx_ + chunksize])
            result = list(exc.map(process_fn, paths[idx_ : idx_ + chunksize]))
            progress_bar.set_description("# Processed: {:d}".format(idx_))


    print("Time Taken: {:.3f}".format(time() - start))


# TODO: Add parser to generate each split
if __name__ == "__main__":
    chunksize = cpus = 4
    # cpus = 1
    start_idx = 0

    BASE_DIR = "/DATA/Users/amahmood/braintyp/"
    DATASET = sys.argv[1]
    split = "train"

    contrast_experiment = False
    contrast_multiples = [110, 120, 130, 140]

    if DATASET == "LESION":
        SAVE_DIR = os.path.join(BASE_DIR, "lesion-hc")
        paths = get_lesionpaths()
    elif DATASET == "TUMOR":
        SAVE_DIR = os.path.join(BASE_DIR, "tumor")
        paths = get_bratspaths()
    elif DATASET == "IBIS":
        SAVE_DIR = os.path.join(BASE_DIR, "ibis")
        paths = get_ibispaths()
    elif DATASET == "HCPD":
        SAVE_DIR = os.path.join(BASE_DIR, "hcpd")
        paths = get_hcpdpaths()
    else:
        SAVE_DIR = os.path.join(BASE_DIR, "processed")
        paths = get_abcdpaths(split)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    process_fn = lesion_preprocessor if DATASET == "LESION" else preprocessor

    if contrast_experiment:
        for contrast in contrast_multiples:
            print("SAVING HIGH CONTRAST LESIONS!!")
            SAVE_DIR = os.path.join(BASE_DIR, f"lesion-c{contrast}")
            os.makedirs(SAVE_DIR, exist_ok=True)
            run(paths, process_fn)

    else:
        run(paths, process_fn)
