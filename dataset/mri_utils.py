import torch
import numpy as np
from monai.transforms.croppad.array import CenterSpatialCrop
from monai.transforms.transform import Randomizable, Transform
from monai.transforms.utils import (
    create_control_grid,
    create_grid,
)
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    ensure_tuple,
    fall_back_tuple,
)

from monai.transforms.spatial.array import *
from typing import *
from monai.transforms import *

"""
Adapted from FastMRI
"""


class TumorGrowthGrid3D(RandDeformGrid):
    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        max_tumor_size: float,
        magnitude_range: Tuple[float, float],
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super(TumorGrowthGrid3D, self).__init__(
            spacing, magnitude_range, as_tensor_output, device
        )
        self.max_tumor_size = max_tumor_size

    def randomize(self) -> None:
        h1, w1, d1 = self.R.randint(-self.max_tumor_size, self.max_tumor_size, size=3)
        h2, w2, d1 = self.R.randint(-self.max_tumor_size, self.max_tumor_size, size=3)
        self.center1 = np.array([h1, w1, d1, 1], dtype=np.float32).reshape(4, 1, 1, 1)
        self.center2 = np.array([h2, w2, d1, 1], dtype=np.float32).reshape(4, 1, 1, 1)
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size: Sequence[int]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: spatial size of the grid.
        """
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(spatial_size, self.spacing)
        self.randomize()
        dist1 = np.sqrt(
            np.sum(np.square(control_grid - self.center1)[:3], axis=0) + 1e-6
        )
        dist2 = np.sqrt(
            np.sum(np.square(control_grid - self.center2)[:3], axis=0) + 1e-6
        )
        deform_mag = self.rand_mag / (dist1 + dist2)
        center = (self.center1 + self.center2) / 2.0
        deform = (control_grid - center)[:3] * deform_mag
        control_grid[: len(spatial_size)] -= deform
        # print(control_grid.shape, deform.shape)
        # print(self.center1, self.center2)

        if self.as_tensor_output:
            control_grid = torch.as_tensor(
                np.ascontiguousarray(control_grid), device=self.device
            )
        return control_grid


class RandTumor(Randomizable, Transform):
    def __init__(
        self,
        spacing: Union[Tuple[float, float], float],
        max_tumor_size: float,
        magnitude_range: Tuple[float, float],
        prob: float = 0.1,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.deform_grid = TumorGrowthGrid3D(
            spacing=spacing,
            max_tumor_size=max_tumor_size,
            magnitude_range=magnitude_range,
            as_tensor_output=True,
            device=device,
        )
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)

        self.spatial_size = spatial_size
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.prob = prob
        self.do_transform = False

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ):
        self.deform_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self) -> None:
        self.do_transform = self.R.rand() < self.prob
        self.deform_grid.randomize()

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        spatial_size: Optional[Union[Tuple[int, int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        self.randomize()
        if self.do_transform:
            grid = self.deform_grid(spatial_size=sp_size)
            grid = torch.nn.functional.interpolate(
                input=grid.unsqueeze(0),
                scale_factor=list(ensure_tuple(self.deform_grid.spacing)),
                mode=InterpolateMode.TRILINEAR.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            grid = create_grid(spatial_size=sp_size)
        return self.resampler(
            img,
            grid,
            mode=mode or self.mode,
            padding_mode=padding_mode or self.padding_mode,
        )
