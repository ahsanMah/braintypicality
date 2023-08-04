# Run in slicer using exec(open(fname).read())

import os
import re
import glob
from time import time

# BASEDIR = "/Users/smaug/mnt/JANUS/"
BASEDIR = "/"
SAVEDIR = f"{BASEDIR}/ASD/ahsan_projects/lesion_samples/"

def load_volumes(path):
    slicer.mrmlScene.Clear(0)

    t1_path = os.path.join(BASEDIR, path)
    t2_path = os.path.realpath(path.replace("T1", "T2"))
    t2_path = f"{BASEDIR}/{t2_path}"

    t1_volume_node = slicer.util.loadVolume(t1_path)
    t2_volume_node = slicer.util.loadVolume(t2_path)
    print(f"Loaded volumes from {os.path.dirname(t1_path)}")

    return t1_volume_node, t2_volume_node


def generate_lesions(sample_path):
    """
    Will run the lesion simulator on the input volumes
    Run this in slicer using exec(open(fname).read())
    Can only run this in Slicer Python
    """
    import slicer
    import MSLesionSimulator

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
    msLesionSimulatorWidget.setReturnOriginalSpaceBooleanWidget.setChecked(True)
    msLesionSimulatorWidget.setNumberOfThreadsWidget.setValue(9)
    msLesionSimulatorWidget.setPercSamplingQWidget.setValue(0.1)

    #! MAKE SURE THAT THE APPROPRIATE PARAMS ARE SET IN THE GUI
    msLesionSimulatorWidget.onApplyButton()

    # Save the lesion label map and the lesioned volumes
    
    sample_id = os.path.basename(sample_path).split("_")[0]
    savedir = f"{SAVEDIR}/{sample_id}/"
    os.makedirs(savedir, exist_ok=True)

    lesionLabelNode = slicer.util.getNode("T1_lesion_label")
    slicer.util.saveNode(lesionLabelNode, f"{savedir}/lesion_label.nrrd")
    print("Saved lesion label map")
    slicer.util.saveNode(t1_volume_node, f"{savedir}/lesioned_test.nrrd")
    slicer.util.saveNode(t2_volume_node, f'{savedir}/T2_lesioned_test.nrrd')
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


def lesion_preprocessing_runner(path, dataset="IBIS"):
    import tensorflow as tf
    import ants
    from generate_mri import register_and_match

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

    if dataset == "ABCD":
        R = re.compile(r"Data\/sub-(.*)\/ses-")
        subject_id = R.search(path).group(1)
        t1_path = path
        t2_path = path.replace("T1w", "T2w")
    elif dataset == "IBIS":
        subject_id, t1_path = path
        subject_id = "IBIS_" + subject_id
        t2_path = t1_path.replace("T1w", "T2w")
    elif dataset == "HCPD":
        subject_id, t1_path = path
        t2_path = t1_path.replace("T1w_", "T2w_")
    else:
        raise NotImplementedError

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


def preprocessing_pipeline(chunksize=4):
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor

    start_idx = 0
    start = time()

    paths = get_inlier_ibis_paths()

    progress_bar = tqdm(
        range(0, len(paths), chunksize),
        total=len(paths) // chunksize,
        initial=0,
        desc="# Processed: ?",
    )

    with ProcessPoolExecutor(max_workers=chunksize) as exc:
        for idx in progress_bar:
            idx_ = idx + start_idx
            result = list(
                exc.map(lesion_preprocessing_runner, paths[idx_ : idx_ + chunksize])
            )
            progress_bar.set_description("# Processed: {:d}".format(idx_))

    print("Time Taken: {:.3f}".format(time() - start))


def lesion_generation_pipeline():
    
    start = time()

    processed_paths = glob.glob(
        f"/ASD/ahsan_projects/lesion_samples/preprocessed/IBIS_*/*_T1.nrrd"
    )

    for path in processed_paths:
        generate_lesions(path)

    print("Time Taken: {:.3f}".format(time() - start))


if __name__ == "__main__":
    preprocessing_pipeline()
    # run()
