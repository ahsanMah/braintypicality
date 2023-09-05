# Run in slicer using exec(open(fname).read())

import functools
import glob
import os
import re
from time import time

# BASEDIR = "/Users/smaug/mnt/JANUS/"
BASEDIR = "/"
SAVEDIR = f"{BASEDIR}/ASD/ahsan_projects/lesion_samples/"


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

def load_volumes(path):
    slicer.mrmlScene.Clear(0)

    t1_path = os.path.join(BASEDIR, path)
    t2_path = os.path.realpath(path.replace("T1", "T2"))
    t2_path = f"{BASEDIR}/{t2_path}"

    t1_volume_node = slicer.util.loadVolume(t1_path)
    t2_volume_node = slicer.util.loadVolume(t2_path)
    print(f"Loaded volumes from {os.path.dirname(t1_path)}")

    return t1_volume_node, t2_volume_node


def generate_lesions(sample_path, lesion_load=10):
    """
    Will run the lesion simulator on the input volumes
    Run this in slicer using exec(open(fname).read())
    Can only run this in Slicer Python
    """
    import MSLesionSimulator
    import slicer

    print("Running lesion runner script...")
    # for i in range(1):
    slicer.mrmlScene.Clear(0)

    t1_volume_node, t2_volume_node = load_volumes(sample_path)

    # Assuming MSLesionSimulatorWidget is a defined widget in that module
    moduleWidget = slicer.modules.mslesionsimulator.widgetRepresentation()

    # Get the instance of the MSLesionSimulatorWidget class
    msLesionSimulatorWidget = moduleWidget.self()

    # Set the volume node in the selector
    msLesionSimulatorWidget.inputT1Selector.setCurrentNode(t1_volume_node)
    msLesionSimulatorWidget.inputT2Selector.setCurrentNode(t2_volume_node)
    print("Populated the input volume selectors")

    # Setting some params
    msLesionSimulatorWidget.setIsBETBooleanWidget.setChecked(True)
    msLesionSimulatorWidget.setReturnOriginalSpaceBooleanWidget.setChecked(True)
    msLesionSimulatorWidget.setNumberOfThreadsWidget.setValue(8)
    msLesionSimulatorWidget.setPercSamplingQWidget.setValue(0.1)

    msLesionSimulatorWidget.lesionLoadSliderWidget.value = lesion_load

    #! MAKE SURE THAT THE APPROPRIATE PARAMS ARE SET IN THE GUI
    msLesionSimulatorWidget.onApplyButton()

    # Save the lesion label map and the lesioned volumes

    sample_id = os.path.basename(sample_path).split("_")[1]
    savedir = f"{SAVEDIR}/lesion_load_{lesion_load}/{sample_id}/"
    os.makedirs(savedir, exist_ok=True)

    lesionLabelNode = slicer.util.getNode("T1_lesion_label")
    slicer.util.saveNode(lesionLabelNode, f"{savedir}/{sample_id}_label.nrrd")
    print("Saved lesion label map")
    slicer.util.saveNode(t1_volume_node, f"{savedir}/{sample_id}_T1.nrrd")
    slicer.util.saveNode(t2_volume_node, f"{savedir}/{sample_id}_T2.nrrd")
    print("Saved lesioned volumes")

    print("Finished lesion runner script!")

    return


def get_inlier_ibis_paths():
    R = re.compile(r"stx_(\d*)_VSA*_*")

    paths = glob.glob(
        f"{BASEDIR}/ASD/Autism/IBIS/Proc_Data/*/VSA*/mri/registered_stx/sMRI/*T1w.nrrd"
    )
    print("Found:", len(list(paths)))

    splits_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(splits_dir, "ibis_inlier_keys.txt"), "r") as f:
        subject_ids = set([x.strip() for x in f.readlines()])

    id_paths = []
    for path in paths:
        match = R.search(path)
        sub_id = match.group(1)
        if sub_id not in subject_ids:
            continue
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths


def get_inlier_abcd_hcpd_paths():
    # abcd_dir = "/BEE/Connectome/ABCD/"
    abcd_dir = "/DATA/"
    abcd_paths = glob.glob(f"{abcd_dir}/ImageData/Data/*/ses-baselineYear1Arm1/anat/*T1w.nii.gz")

    hcpd_paths = glob.glob("/UTexas/HCP/HCPD/fmriresults01/*_V1_MR/T1w/T1w_acpc_dc.nii.gz")
    clean = lambda x: x.strip().replace("_", "")

    curr_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{curr_dir}/test_keys.txt", "r") as f:
        inlier_keys = set([clean(x) for x in f.readlines()])

    inlier_paths = []

    R = get_matcher("ABCD")
    for path in abcd_paths:
        match = R.search(path)
        sub_id = match.group(1)

        if sub_id in inlier_keys:
            inlier_paths.append((sub_id, path))

    R = get_matcher("HCPD")
    for path in hcpd_paths:
        match = R.search(path)
        sub_id = match.group(1)

        if sub_id in inlier_keys:
            inlier_paths.append((sub_id, path))
    
    # print(inlier_paths)
    print("Collected:", len(inlier_paths))

    return inlier_paths


def lesion_preprocessing_runner(path, dataset="ABCD"):
    import ants
    import tensorflow as tf
    from generate_mri import register_and_match

    # dataset = DATASET
    cache_dir = "/ASD/ahsan_projects/braintypicality/dataset/template_cache/"
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

    subject_id, t1_path = path
    t2_path = t1_path.replace("T1w", "T2w")
    subject_id = f"{dataset}_{subject_id}"

    t1_img = ants.image_read(t1_path)
    t2_img = ants.image_read(t2_path)

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

    # Save outputs
    dirname = f"/ASD/ahsan_projects/lesion_samples/preprocessed/{subject_id}/"
    os.makedirs(dirname, exist_ok=True)
    t1_img.to_filename(f"{dirname}/{subject_id}_T1.nrrd")
    t2_img.to_filename(f"{dirname}/{subject_id}_T2.nrrd")

    return


def preprocessing_pipeline(dataset, chunksize=4):
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    start_idx = 0
    start = time()

    if dataset == "IBIS":
        paths = get_inlier_ibis_paths()
    elif dataset == "ABCD":
        paths = get_inlier_abcd_hcpd_paths()
    else:
        raise NotImplementedError

    progress_bar = tqdm(
        range(0, len(paths), chunksize),
        total=len(paths) // chunksize,
        initial=0,
        desc="# Processed: ?",
    )
    # global DATASET
    # DATASET = dataset

    # progress_bar = range(0, len(paths), chunksize)
    runner = functools.partial(lesion_preprocessing_runner, dataset=dataset)

    with ProcessPoolExecutor(max_workers=chunksize) as exc:
        for idx in progress_bar:
            idx_ = idx + start_idx
            result = list(
                exc.map(runner, paths[idx_ : idx_ + chunksize])
            )
            progress_bar.set_description("# Processed: {:d}".format(idx_))

    print("Time Taken: {:.3f}".format(time() - start))


def lesion_generation_pipeline(lesion_load=10, dataset="IBIS"):
    assert dataset in ["IBIS", "ABCD"]

    start = time()

    processed_paths = glob.glob(
        f"/ASD/ahsan_projects/lesion_samples/preprocessed/{dataset}_*/*_T1.nrrd"
    )

    for path in processed_paths:
        generate_lesions(path, lesion_load)

    print("Time Taken: {:.3f}".format(time() - start))


# TODO: Add functionality to enhance the lesioned regions
def postprocessing_pipeline(lesion_load=10, dataset="IBIS"):
    import ants
    from tqdm import tqdm

    start = time()

    lesion_sample_paths = glob.glob(f"{SAVEDIR}/lesion_load_{lesion_load}/{dataset}_*")

    progress_bar = tqdm(
        range(0, len(lesion_sample_paths)),
        initial=0,
        desc="# Processed: ?",
    )

    for idx in progress_bar:
        path = lesion_sample_paths[idx]
        sample_id = os.path.basename(path)
        # print(f"Processing {sample_id}")
        t1_path = f"{path}/{sample_id}_T1.nrrd"
        t2_path = f"{path}/{sample_id}_T2.nrrd"
        label_path = f"{path}/{sample_id}_label.nrrd"

        t1_img = ants.image_read(t1_path)
        t2_img = ants.image_read(t2_path)
        label_img = ants.image_read(label_path)
        combined_img = ants.merge_channels([t1_img, t2_img])
        combined_img.to_filename(f"{path}/{sample_id}.nii.gz")
        label_img.to_filename(f"{path}/{sample_id}_label.nii.gz")

        progress_bar.set_description("# Processed: {:d}".format(idx))

    print("Time Taken: {:.3f}".format(time() - start))


'''
lesion_generation_pipeline can only be run in slicer with the MS Lesion Simulator extension
pre and post processing should be run on normal pythpon environment
'''
if __name__ == "__main__":
    # preprocessing_pipeline("ABCD")
    # lesion_generation_pipeline(lesion_load=20, dataset="ABCD")
    # postprocessing_pipeline(lesion_load=20)
