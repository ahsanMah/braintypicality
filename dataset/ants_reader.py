from typing import List, Sequence, Union

import ants
import nibabel as nib
from monai.config import DtypeLike
from monai.data import NibabelReader
from monai.data.utils import correct_nifti_header_if_necessary
from monai.transforms import *
from monai.utils import ensure_tuple

from dataset.generate_mri import register_and_match


class ANTSReader(NibabelReader):
    """
    Use this class as a generic loader for any nifti file using the ANTs library.
    Loading as an ANTs datatype allows us to use the ANTSPyNet built in functions
    e.g. auto brain masking
    """

    def __init__(
        self,
        dtype: DtypeLike = np.float32,
        modality: Union[Sequence[str], str] = "t1",
        register_to_template: bool = True,
        extract_brain_mask: bool = False,
        match_intensity: bool = False,
        **kwargs,
    ):

        super().__init__()
        self.dtype = dtype
        self.modality = "t1"
        self.register_to_template = register_to_template
        self.extract_brain_mask = extract_brain_mask
        self.match_intensity = match_intensity
        self.kwargs = kwargs

    def read(self, data: Union[Sequence[str], str], **kwargs):
        """
        Read image data from specified file or files.
        Note that the returned object is Nibabel image object or list of Nibabel image objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `nibabel.load` API, will override `self.kwargs` for existing keys.
                More details about available args:
                https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py

        """
        img_: List[Nifti1Image] = []

        filenames: Sequence[str] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)

        for name in filenames:
            img = nib.load(name, **kwargs_)

            # This is used to laod the appropriate channels from BRATs dataset
            t1_img = ants.from_nibabel(img.slicer[..., 2])
            # print(t1_img)
            t2_img = ants.from_nibabel(img.slicer[..., 3])
            _mask = t1_img != 0

            preproc_img = ants.merge_channels(
                [
                    register_and_match(
                        t1_img,
                        _mask,
                        modality="t1",
                        verbose=False,
                    ),
                    register_and_match(
                        t2_img,
                        _mask,
                        modality="t2",
                        verbose=False,
                    ),
                ]
            )

            img = preproc_img.to_nibabel()
            img = correct_nifti_header_if_necessary(img)
            img_.append(img)

        return img_ if len(filenames) > 1 else img_[0]


if __name__ == "__main__":

    loader = LoadImaged("image")
    loader.register(ANTSReader())

    img_transform = Compose(
        [
            loader,
            SqueezeDimd("image", dim=3),
            AsChannelFirstd("image"),
            SpatialCropd("image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]),
            DivisiblePadd("image", k=8),
        ]
    )

    # Load ood dataset
    # ood_ds = DecathlonDataset(
    #     root_dir=config.data.tumor_dir_path,
    #     task="Task01_BrainTumour",
    #     section="validation",
    #     transform=img_transform,
    #     val_frac=0.01,
    #     cache_rate=1.0,
    #     num_workers=4,
    #     download=True,
    # )
