# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import os
import glob
import torch
import ants
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# import tensorflow_addons as tfa
import matplotlib.pyplot as plt


from monai.data import CacheDataset, DataLoader, ArrayDataset, PersistentDataset
from monai.transforms import *
from dataset.mri_utils import RandTumor


def ants_plot_scores(x, fname):
    """
    Plot scores for a single sample
    Expects (num_scores, h,w,d)
    """
    import numpy as np
    from PIL import Image

    plot_ax = 2
    n = x.shape[0]
    # c = x.shape[1]
    sz = x.shape[plot_ax]
    nslices = 5
    slices = np.linspace(sz // 5, 3 * sz // 5, nslices, dtype=np.int)
    # x_imgs = [[ants.from_numpy(sample)] * nslices for sample in x]
    x_imgs = [
        [ants.from_numpy(sample[..., i % 2])] * nslices for i, sample in enumerate(x)
    ]
    ants.plot_grid(
        x_imgs,
        slices=np.tile(slices, [n, 1]),
        dpi=100,
        cmap="Reds",
        axes=plot_ax,
        filename=fname,
    )
    # im = np.asarray(Image.open("tmp.png"))
    # im = tf.expand_dims(im, 0)
    return


def plot_slices(x, fname, channels_first=False):
    # print("Before plotting:", x.shape)

    if channels_first:
        if isinstance(x, np.ndarray):
            x = np.transpose(x, axes=(0, 2, 3, 4, 1))

        if isinstance(x, torch.Tensor):
            x = x.permute(0, 2, 3, 4, 1).detach().cpu().numpy()

    # Get alternating channels per sample
    c = x.shape[-1]
    x_imgs = [ants.from_numpy(sample[..., i % c]) for i, sample in enumerate(x)]
    ants.plot_ortho_stack(
        x_imgs,
        orient_labels=False,
        dpi=100,
        filename=fname,
        transparent=True,
        crop=True,
    )

    return


def get_channel_selector(config):
    c = config.data.select_channel
    if c > -1:
        return lambda x: torch.unsqueeze(x[:, c, ...], dim=1)
    else:
        return lambda x: x


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""

    # Optionally select channels
    n = config.data.num_channels

    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC,
    )
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(
        image, [h, w], antialias=True, method=tf.image.ResizeMethod.BICUBIC
    )


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False, ood_eval=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """

    if config.colab:
        config.data.dir_path = config.data.colab_path
        config.data.splits_path = config.data.colab_splits_path
        config.data.tumor_dir_path = config.data.colab_tumor_path

    # Compute batch size for this worker.
    batch_size = (
        config.training.batch_size if not evaluation else config.eval.batch_size
    )
    # if batch_size % jax.device_count() != 0:
    #     raise ValueError(
    #         f"Batch sizes ({batch_size} must be divided by"
    #         f"the number of devices ({jax.device_count()})"
    #     )

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 100  # 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    if config.data.dataset == "BRAIN":
        dataset_dir = config.data.dir_path
        splits_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "dataset"
        )  # config.data.splits_path
        clean = lambda x: x.strip().replace("_", "")
        # print("Dir for keys:", splits_dir)
        filenames = {}
        for split in ["train", "val", "test"]:
            with open(os.path.join(splits_dir, f"{split}_keys.txt"), "r") as f:
                filenames[split] = [clean(x) for x in f.readlines()]

        val_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")}
            for x in filenames["val"]
        ]

        train_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")}
            for x in filenames["train"]
        ]

        test_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")}
            for x in filenames["test"]
        ]

        # ood_file_list = [
        #     {"image": os.path.join(dataset_dir, f"{x}.nii.gz")}
        #     for x in filenames["ood"]
        # ]

        CACHE_RATE = config.data.cache_rate
        spacing = [config.data.spacing_pix_dim] * 3
        cache_dir_name = "/tmp/monai_brains/train"

        if config.data.spacing_pix_dim > 1.0:
            cache_dir_name += f"_downsample_{config.data.spacing_pix_dim}"
            if CACHE_RATE > 0.0:
                print("Using cache dir:", cache_dir_name)

        train_transform = Compose(
            [
                LoadImaged("image", image_only=True),
                SqueezeDimd("image", dim=3),
                AsChannelFirstd("image"),
                SpatialCropd("image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]),
                Spacingd("image", pixdim=spacing),
                DivisiblePadd("image", k=32),
                # RandStdShiftIntensityd("image", (-0.1, 0.1)), #-0.05->5
                # RandScaleIntensityd("image", (-0.1, 0.1)),
                RandStdShiftIntensityd("image", (-0.05, 0.05)),
                RandScaleIntensityd("image", (-0.05, 0.05)),
                RandHistogramShiftd("image", num_control_points=[3, 5]),
                # RandAxisFlipd("image", 0.5),
                RandFlipd("image", prob=0.5, spatial_axis=0),
                RandAffined(
                    "image",
                    prob=0.1,
                    rotate_range=[0.03, 0.03, 0.03],
                    translate_range=3,
                ),
                ScaleIntensityd("image", minv=0, maxv=1.0),
            ]
        )

        val_transform = Compose(
            [
                LoadImaged("image", image_only=True),
                SqueezeDimd("image", dim=3),
                AsChannelFirstd("image"),
                SpatialCropd("image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]),
                Spacingd("image", pixdim=spacing),
                DivisiblePadd("image", k=32),
                ScaleIntensityd("image", minv=0, maxv=1.0),
            ]
        )

        if not evaluation:
            train_ds = PersistentDataset(
                train_file_list,
                transform=train_transform,
                # cache_rate=CACHE_RATE,
                # num_workers=4,
                # progress=False,
                cache_dir=cache_dir_name,
            )

            eval_ds = CacheDataset(
                val_file_list,
                transform=val_transform,
                cache_rate=CACHE_RATE,
                num_workers=4,
                progress=False,
            )

        elif not ood_eval:
            train_ds = CacheDataset(
                train_file_list,
                transform=val_transform,
                cache_rate=CACHE_RATE,
                num_workers=4,
            )

            eval_ds = CacheDataset(
                val_file_list,
                transform=val_transform,
                cache_rate=CACHE_RATE,
                num_workers=4,
            )

        else:  # evaluation AND ood_eval
            train_ds = None
            inlier_file_list = test_file_list
            img_transform = val_transform

            # Generate OOD samples by adding "tumors" to center
            # i.e. compute random grid deformations
            if config.data.gen_ood:
                deformer = RandTumor(
                    spacing=1.0,
                    max_tumor_size=5.0 / config.data.spacing_pix_dim,
                    magnitude_range=(
                        5.0 / config.data.spacing_pix_dim,
                        15.0 / config.data.spacing_pix_dim,
                    ),
                    prob=1.0,
                    spatial_size=config.data.image_size,  # [168, 200, 152],
                    padding_mode="zeros",
                )

                deformer.set_random_state(seed=0)

                ood_transform = Compose(
                    [
                        LoadImaged("image", image_only=True),
                        SqueezeDimd("image", dim=3),
                        AsChannelFirstd("image"),
                        SpatialCropd(
                            "image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]
                        ),
                        Spacingd("image", pixdim=spacing),
                        DivisiblePadd("image", k=8),
                        RandLambdad("image", deformer),
                    ]
                )

                ood_file_list = test_file_list
                img_transform = ood_transform

            elif config.data.ood_ds in ["ASD", "DS-SA"]:
                filenames = {}

                prefix = config.data.ood_ds.lower()

                for split in ["inlier", "outlier"]:
                    with open(os.path.join(splits_dir, f"{prefix}_{split}_keys.txt"), "r") as f:
                        filenames[split] = [x.strip() for x in f.readlines()]

                inlier_file_list = [
                    {"image": os.path.join(dataset_dir, "..", "ibis", f"{x}.nii.gz")}
                    for x in filenames["inlier"]
                ]

                ood_file_list = [
                    {"image": os.path.join(dataset_dir, "..","ibis", f"{x}.nii.gz")}
                    for x in filenames["outlier"]
                ]

            elif "LESION" in config.data.ood_ds:
                suffix = ""
                if "-" in config.data.ood_ds:
                    _, suffix = config.data.ood_ds.split("-")
                    suffix = "-" + suffix

                dirname = "lesion" + suffix

                ood_file_list = [
                    {"image": x}
                    for x in glob.glob(os.path.join(dataset_dir, "..", dirname, "*"))
                ]
                print("Collected samples:", len(ood_file_list), "from", dirname)

            # Load either real or generated ood samples
            # Defaults to ABCD test/ood data
            train_ds = CacheDataset(
                inlier_file_list,
                transform=val_transform,
                cache_rate=CACHE_RATE * 0,
                num_workers=4,
            )

            eval_ds = CacheDataset(
                ood_file_list,
                transform=img_transform,
                cache_rate=CACHE_RATE * 0,
                num_workers=4,
            )

    else:
        raise NotImplementedError(f"Dataset {config.data.dataset} not yet supported.")

    def make_generator(ds):

        single_channel = config.data.num_channels == 1

        def tf_gen_img():
            for x in ds:
                img = x["image"]
                if single_channel:
                    img = img[:1, ...]
                yield img

        return tf_gen_img

    def create_tfds_dataset(data_loader, val=False):

        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)

        output_type = tf.float32
        tensor_sz = np.array(config.data.image_size)  # / config.data.spacing_pix_dim
        img_h, img_w, img_d = tensor_sz
        c = config.data.num_channels
        output_shape = tf.TensorShape([c, img_h, img_w, img_d])

        ds = tf.data.Dataset.from_generator(
            make_generator(data_loader),
            output_type,
            output_shape,
            # output_signature=(tf.TensorSpec(shape=(img_h, img_w, c), dtype=tf.float32)),
        )

        # ds = ds.cache()
        if not val:
            ds = ds.repeat(count=num_epochs)
            # ds = ds.shuffle(shuffle_buffer_size)

        ds = ds.batch(
            config.eval.batch_size if val else batch_size, drop_remainder=False
        )

        return ds.prefetch(prefetch_size)

    if config.data.dataset in ["BRAIN", "TUMOR"]:
        if train_ds:
            if config.data.as_tfds:
                train_ds = create_tfds_dataset(train_ds)
            else:
                train_ds = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=evaluation == False,
                    num_workers=4,
                    pin_memory=False,
                )
        if eval_ds:
            if config.data.as_tfds:
                eval_ds = create_tfds_dataset(eval_ds, val=True)
            else:
                eval_ds = DataLoader(
                    eval_ds,
                    batch_size=config.eval.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=False,
                )
        dataset_builder = None

    #### Test if loader worked

    # for x in train_ds:
    #     # print("Shape:", x["image"].shape)
    #     x = x["image"]
    #     print("Shape:", x.shape)
    #     # print(x["image"].numpy().max())
    #     # q = np.quantile(x["image"].numpy(), 0.999)
    #     # plt.imshow(x["image"][0,0,128,], vmax=q)
    #     plot_slices(
    #         x.numpy(), fname=f"{config.data.dataset}_sample.png", channels_first=True
    #     )
    #     break
    # exit()

    return train_ds, eval_ds, dataset_builder
