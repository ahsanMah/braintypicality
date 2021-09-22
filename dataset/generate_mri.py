import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob
import re, pickle
import time
import ants, antspynet
import numpy as np

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from time import time

# For Docker Images
DATA_DIR = "/DATA/"
SAVE_DIR = "/DATA/Users/amahmood/braintyp/"
CACHE_DIR = os.getcwd() + "/template_cache"

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


def extract_brain_mask(image, antsxnet_cache_directory=None, verbose=True):
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
        modality="t1",
        antsxnet_cache_directory=antsxnet_cache_directory,
        verbose=verbose,
    )
    mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
    mask = ants.morphology(mask, "close", 6).iMath_fill_holes()

    return mask


def register_and_match(
    image,
    mask,
    truncate_intensity=(0.01, 0.99),
    modality="t1",
    template_transform_type="Rigid",
    antsxnet_cache_directory=None,
    verbose=True,
):

    """
    Basic preprocessing pipeline for T1-weighted brain MRI adapted from AntsPyNet
    https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/brain_extraction.py

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

    # Template normalization
    transforms = None
    if modality == "t1":
        template_image = t1_ref_img
    else:
        template_image = t2_ref_img

    template_brain_image = template_image * ref_img_mask

    preprocessed_brain_image = preprocessed_image * mask
    registration = ants.registration(
        fixed=template_brain_image,
        moving=preprocessed_brain_image,
        type_of_transform=template_transform_type,
        verbose=verbose,
    )
    transforms = dict(
        fwdtransforms=registration["fwdtransforms"],
        invtransforms=registration["invtransforms"],
    )

    preprocessed_image = ants.apply_transforms(
        fixed=template_image,
        moving=preprocessed_image,
        transformlist=registration["fwdtransforms"],
        interpolator="linear",
        verbose=verbose,
    )
    mask = ants.apply_transforms(
        fixed=template_image,
        moving=mask,
        transformlist=registration["fwdtransforms"],
        interpolator="genericLabel",
        verbose=verbose,
    )

    # Do bias correction
    bias_field = None

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
    preprocessed_image = preprocessed_image * mask
    preprocessed_image = ants.utils.histogram_match_image(
        preprocessed_image,
        template_brain_image,
        number_of_histogram_bins=128,
        number_of_match_points=10,
    )
    # Min max norm
    preprocessed_image = (preprocessed_image - preprocessed_image.min()) / (
        preprocessed_image.max() - preprocessed_image.min()
    )

    return preprocessed_image


def preprocessor(sample):

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
    t1_path = path
    t2_path = path.replace("T1w", "T2w")

    t1_img = ants.image_read(t1_path)
    t2_img = ants.image_read(t2_path)

    _mask = extract_brain_mask(
        t1_img, antsxnet_cache_directory=CACHE_DIR, verbose=False
    )
    preproc_img = ants.merge_channels(
        [
            register_and_match(
                t1_img,
                _mask,
                modality="t1",
                antsxnet_cache_directory=CACHE_DIR,
                verbose=False,
            ),
            register_and_match(
                t2_img,
                _mask,
                modality="t2",
                antsxnet_cache_directory=CACHE_DIR,
                verbose=False,
            ),
        ]
    )

    fname = os.path.join(SAVE_DIR, f"{subject_id}.nii.gz")
    preproc_img.to_filename(fname)

    return subject_id


def get_matcher(dataset):

    # ABCD adult matcher
    if dataset == "ABCD":
        return re.compile(r"Data\/sub-(.*)\/ses-")  # NDAR..?

    matcher = r"neo-\d{4}-\d(-\d)?"

    if dataset == "CONTE2":
        return re.compile(matcher)

    return re.compile("(" + matcher + ")")


def get_abcdpaths(split="train"):

    assert split in ["train", "val"]

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


# TODO: Add parser to generate each split

start = time()
chunksize = 4
cpus = 4
start_idx = 0

split = "test"
SAVE_DIR = os.path.join(SAVE_DIR, split)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

paths = get_abcdpaths(split)
# process_fn = make_processor(split)


progress_bar = tqdm(
    range(0, len(paths), chunksize),
    total=len(paths) // chunksize,
    initial=0,
    desc="# Processed: ?",
)

# for idx in progress_bar:
#     preprocessor(paths[idx])
#     break

with ProcessPoolExecutor(max_workers=cpus) as exc:
    for idx in progress_bar:
        idx_ = idx + start_idx
        result = list(exc.map(preprocessor, paths[idx_ : idx_ + chunksize]))
        progress_bar.set_description("# Processed: {:d}".format(idx_))

print("Time Taken: {:.3f}".format(time() - start))
# print(result)
