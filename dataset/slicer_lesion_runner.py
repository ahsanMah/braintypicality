# Run in slicer using exec(open(fname).read())

import os
import re
import glob
import slicer
import MSLesionSimulator

BASEDIR = "/Users/smaug/mnt/JANUS/"


def load_volumes(sample_id, path):

    slicer.mrmlScene.Clear(0)

    t1_path = os.path.join(BASEDIR, path)
    t2_path = os.path.realpath(path.replace("T1w", "T2w"))
    t2_path = f"{BASEDIR}/{t2_path}"

    t1_volume_node = slicer.util.loadVolume(t1_path)
    t2_volume_node = slicer.util.loadVolume(t2_path)
    print(f"Loaded volumes from {os.path.dirname(t1_path)}")

    return t1_volume_node, t2_volume_node


def run():
    print("Running lesion runner script...")
    # for i in range(1):
    slicer.mrmlScene.Clear(0)

    sample_id = 108372
    t1_volume_node, t2_volume_node = load_volumes(sample_id, path)

    # Assuming MSLesionSimulatorWidget is a defined widget in that module
    moduleWidget = slicer.modules.mslesionsimulator.widgetRepresentation()

    # Get the instance of the MSLesionSimulatorWidget class
    msLesionSimulatorWidget = moduleWidget.self()

    # Set the volume node in the selector
    msLesionSimulatorWidget.inputT1Selector.setCurrentNode(t1_volume_node)
    # msLesionSimulatorWidget.inputT2Selector.setCurrentNode(t2_volume_node)
    print("Populated the input volume selectors")

    # Setting some params
    msLesionSimulatorWidget.setReturnOriginalSpaceBooleanWidget.setChecked(True)
    msLesionSimulatorWidget.setNumberOfThreadsWidget.setValue(9)
    msLesionSimulatorWidget.setPercSamplingQWidget.setValue(0.1)

    #! MAKE SURE THAT THE APPROPRIATE PARAMS ARE SET IN THE GUI
    msLesionSimulatorWidget.onApplyButton()

    # Save the lesion label map and the lesioned volumes

    savedir = f"{BASEDIR}/ASD/ahsan_projects/lesion_samples/{sample_id}/"
    lesionLabelNode = slicer.util.getNode("T1_lesion_label")

    slicer.util.saveNode(lesionLabelNode, f"{savedir}/lesion_label.nrrd")
    print("Saved lesion label map")
    slicer.util.saveNode(t1_volume_node, f"{savedir}/lesioned_test.nrrd")
    # slicer.util.saveNode(t2_volume_node, f'{savedir}/T2_lesioned_test.nrrd')
    # print("Saved lesioned volumes")

    print("Finished lesion runner script!")

    return


def get_ibispaths():
    R = re.compile(r"stx_(\d*)_VSA*_*")

    paths = glob.glob(
        f"{BASEDIR}/ASD/Autism/IBIS/Proc_Data/*/VSA*/mri/registered_stx/sMRI/*T1w.nrrd"
    )
    print("Found:", len(list(paths)))

    splits_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "dataset")
    with open(os.path.join(splits_dir, "ibis_inlier_keys.txt"), "r") as f:
        subject_ids = set([x.strip() for x in f.readlines()])

    id_paths = {}
    for path in paths:
        match = R.search(path)
        sub_id = match.group(1)
        if sub_id not in subject_ids:
            continue
        id_paths["sub_id"] = path

    print("Collected:", len(id_paths))

    return id_paths
