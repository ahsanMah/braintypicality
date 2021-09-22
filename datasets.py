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
from tensorflow_datasets.core import dataset_info
import os
import glob
import jax
import torch
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import matplotlib.pyplot as plt


from monai.data import CacheDataset, DataLoader, ArrayDataset
from monai.transforms import *


def plot_slices(x):

    if isinstance(x, torch.Tensor):
        x = x.permute(0, 2, 3, 4, 1).detach().cpu().numpy()

    plt.subplots(3, 3, figsize=(8, 8))
    s = x.shape[2] // 16
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[0, :, (i + 2) * s, :, 0], cmap="gray")
    plt.tight_layout()
    plt.show()
    return


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
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
    # Compute batch size for this worker.
    batch_size = (
        config.training.batch_size if not evaluation else config.eval.batch_size
    )
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch sizes ({batch_size} must be divided by"
            f"the number of devices ({jax.device_count()})"
        )

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 100  # 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    # TODO: Add appropriate OOD evaluation sets for other datasets
    if ood_eval and config.data.dataset in [
        "CIFAR10",
        "SVHN",
        "CELEBA",
        "LSUN",
        "FFHQ",
        "CelebAHQ",
    ]:
        raise NotImplementedError(
            f"OOD evaluation for dataset {config.data.dataset} not yet supported."
        )

    # Create dataset builders for each dataset.
    if config.data.dataset == "CIFAR10":
        dataset_builder = tfds.builder("cifar10")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size], antialias=True
            )

    elif config.data.dataset == "SVHN":
        dataset_builder = tfds.builder("svhn_cropped")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size], antialias=True
            )

    elif config.data.dataset == "CELEBA":
        dataset_builder = tfds.builder("celeb_a")
        train_split_name = "train"
        eval_split_name = "validation"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = central_crop(img, 140)
            img = resize_small(img, config.data.image_size)
            return img

    elif config.data.dataset == "LSUN":
        dataset_builder = tfds.builder(f"lsun/{config.data.category}")
        train_split_name = "train"
        eval_split_name = "validation"

        if config.data.image_size == 128:

            def resize_op(img):
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = resize_small(img, config.data.image_size)
                img = central_crop(img, config.data.image_size)
                return img

        else:

            def resize_op(img):
                img = crop_resize(img, config.data.image_size)
                img = tf.image.convert_image_dtype(img, tf.float32)
                return img

    elif config.data.dataset in ["FFHQ", "CelebAHQ"]:
        dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
        train_split_name = "train"
        eval_split_name = "val"

    elif config.data.dataset == "MVTEC":
        dataset_dir = f"{config.data.dir_path}/{config.data.category}"
        dataset_builder = tfds.ImageFolder(dataset_dir)

        img_sz = config.data.downsample_size

        # Downsample to image size directly as no cropping will be done
        if evaluation:
            img_sz = config.data.image_size

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(
                img,
                [img_sz, img_sz],
                antialias=True,
                method=tf.image.ResizeMethod.LANCZOS5,
            )
            img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
            return img

        def augment_op(img):

            crop_sz = config.data.image_size
            img_sz = config.data.downsample_size

            # Random translate + rotate
            translate_ratio = 0.55 * (crop_sz / img_sz)
            img = tfa.image.rotate(img, tf.random.uniform((1,), 0, np.pi / 2))
            img = tfa.image.translate(
                img,
                tf.random.uniform(
                    (1, 2), -translate_ratio * img_sz, translate_ratio * img_sz
                ),
            )
            img = tf.image.resize_with_crop_or_pad(img, crop_sz, crop_sz)
            img = tf.image.random_hue(img, max_delta=0.05)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_flip_up_down(img)

            return img

        train_split_name = eval_split_name = "train"

        if ood_eval:
            train_split_name = "inlier"
            eval_split_name = "ood"

    elif config.data.dataset == "KNEE":

        rimg_h = rimg_w = config.data.downsample_size

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize_with_pad(
                img,
                rimg_h,
                rimg_h,
                antialias=True,
                method=tf.image.ResizeMethod.LANCZOS5,
            )

            # img = img[15:-15, :]  # Crop high freqs to get square image

            return img

        if config.longleaf:
            dataset_dir = f"{config.data.dir_path_longleaf}"
        else:
            dataset_dir = f"{config.data.dir_path}"

        max_marginal_ratio = config.data.marginal_ratio
        mask_marginals = config.data.mask_marginals
        category = config.data.category
        complex_input = config.data.complex

        train_dir = os.path.join(dataset_dir, "singlecoil_train/")
        val_dir = os.path.join(dataset_dir, "singlecoil_val/")
        test_dir = os.path.join(dataset_dir, "singlecoil_test_v2/")

        img_h, img_w = config.data.original_dimensions
        c = 2 if complex_input else 1

        # if config.mask_marginals:
        #     c += 1

        def normalize(img, complex_input=False, quantile=0.999):

            # Complex tensors are 2D
            if complex_input:
                h = np.quantile(img.reshape(-1, 2), q=quantile, axis=0)
                # l = np.min(img.reshape(-1, 2), axis=0)
                l = np.quantile(img.reshape(-1, 2), q=(1 - quantile) / 10, axis=0)
            else:
                h = np.quantile(img, q=quantile)
                # l = np.min(img)
                l = np.quantile(img, q=(1 - quantile) / 10)

            # Min Max normalize
            img = (img - l) / (h - l)
            img = np.clip(
                img,
                0.0,
                1.0,
            )

            return img

        def make_generator(ds, ood=False):

            if complex_input:
                # Treat input as a 3D tensor (2 channels: real + imag)
                preprocessor = lambda x: np.stack([x.real, x.imag], axis=-1)
                normalizer = lambda x: normalize(x, complex_input=True)
            else:
                preprocessor = lambda x: complex_magnitude(x).numpy()[..., np.newaxis]
                normalizer = lambda x: normalize(x)

            label = 1 if ood else 0

            # TODO: Build complex loader for img

            def tf_gen_img():
                for k, x in ds:
                    img = preprocessor(x)
                    img = normalizer(img)
                    yield img

            def tf_gen_ksp():
                for k, x in ds:
                    img = preprocessor(k)
                    img = normalizer(img)
                    yield img

            if "kspace" == category:
                print(
                    f"Training on {'complex' if complex_input else 'image'} kspace..."
                )
                return tf_gen_ksp

            # Default to target image as category
            print(f"Training on {'complex' if complex_input else 'image'} mri...")
            return tf_gen_img

        def build_ds(datadir, ood=False):

            output_type = tf.float32
            output_shape = tf.TensorShape([img_h, img_w, c])

            dataset = FastKnee(datadir) if not ood_eval else FastKneeTumor(datadir)
            ds = tf.data.Dataset.from_generator(
                make_generator(dataset, ood=ood),
                output_type,
                output_shape,
                # output_signature=(tf.TensorSpec(shape=(img_h, img_w, c), dtype=tf.float32)),
            )

            return ds

        channels = config.data.num_channels

        def np_build_and_apply_random_mask(x):
            # Building mask of random columns to **keep**
            # img_h, img_w, c = x.shape
            bs = x.shape[0]

            rand_ratio = np.random.uniform(
                low=config.data.min_marginal_ratio,
                high=config.data.marginal_ratio,
                size=1,
            )
            n_mask_cols = int(rand_ratio * rimg_w)
            rand_cols = np.random.randint(rimg_w, size=n_mask_cols)

            # We do *not* want to mask out the middle (low) frequencies
            # Keeping 10% of low freq is equivalent to Scenario-30L in activemri paper
            low_freq_cols = np.arange(int(0.45 * rimg_w), img_w - int(0.45 * rimg_w))
            mask = np.zeros((bs, rimg_h, rimg_w, 1), dtype=np.float32)
            mask[..., rand_cols, :] = 1.0
            mask[..., low_freq_cols, :] = 1.0

            # Applying + Appending mask
            x = x * mask
            x = np.concatenate([x, mask], axis=-1)
            return x

        test_slices = 2000
        train_ds = build_ds(train_dir)
        eval_ds = build_ds(val_dir).skip(test_slices)

        # The datsets used to evaluate MSMA
        if ood_eval:
            train_ds = build_ds(val_dir).take(test_slices)
            eval_ds = build_ds(val_dir, ood=True).take(test_slices)

        dataset_builder = train_split_name = eval_split_name = None

    elif config.data.dataset == "BRAIN":
        dataset_dir = config.data.dir_path
        splits_dir = config.data.splits_path
        clean = lambda x: x.strip().replace("_", "")

        filenames = {}
        for split in ["train", "val"]:
            with open(os.path.join(splits_dir, f"{split}_keys.txt"), "r") as f:
                filenames[split] = [clean(x) for x in f.readlines()]

        val_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")}
            for x in filenames["val"]
        ]

        train_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")}
            for x in filenames["val"]
        ]

        img_transform = Compose(
            [
                LoadImaged("image", image_only=True),
                SqueezeDimd("image", dim=3),
                AsChannelFirstd("image"),
                SpatialCropd("image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]),
                DivisiblePadd("image", k=8),
                RandAdjustContrastd("image"),
            ]
        )

        train_ds = CacheDataset(train_file_list[:4], transform=img_transform)
        eval_ds = CacheDataset(val_file_list[:4], transform=img_transform)
    else:
        raise NotImplementedError(f"Dataset {config.data.dataset} not yet supported.")

    # Customize preprocess functions for each dataset.
    if config.data.dataset in ["FFHQ", "CelebAHQ"]:

        def preprocess_fn(d):
            sample = tf.io.parse_single_example(
                d,
                features={
                    "shape": tf.io.FixedLenFeature([3], tf.int64),
                    "data": tf.io.FixedLenFeature([], tf.string),
                },
            )
            data = tf.io.decode_raw(sample["data"], tf.uint8)
            data = tf.reshape(data, sample["shape"])
            data = tf.transpose(data, (1, 2, 0))
            img = tf.image.convert_image_dtype(data, tf.float32)
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (
                    tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
                ) / 256.0
            return dict(image=img, label=None)

    else:

        def preprocess_fn(d):
            """Basic preprocessing function scales data to [0, 1) and randomly flips."""

            if config.data.dataset in ["KNEE"]:
                d = {"image": d}

            img = resize_op(d["image"])
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (
                    tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
                ) / 256.0

            if config.data.dataset == "MVTEC" and not evaluation:
                img = augment_op(img)

            # if config.data.dataset == "KNEE" and config.data.mask_marginals:
            #     img = np_build_and_apply_random_mask(img)

            return dict(image=img, label=d.get("label", None))

    def create_dataset(dataset_builder, split, val=False):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)

        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            if not config.data.dataset == "MVTEC":
                dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(
                split=split, shuffle_files=evaluation, read_config=read_config
            )
        elif dataset_builder not in ["KNEE"]:
            ds = dataset_builder.with_options(dataset_options)
        else:  # dataset_builder is already a TF Dataset
            ds = dataset_builder

        if config.data.dataset == "MVTEC" and not ood_eval:
            val_size = int(0.1 * dataset_builder.info.splits["train"].num_examples)
            if val:
                ds = ds.take(val_size)
            else:  # train split
                ds = ds.skip(val_size)

        ds = ds.cache()
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)

        if config.data.dataset == "KNEE" and config.data.mask_marginals:
            _fn = lambda x: tf.numpy_function(
                func=np_build_and_apply_random_mask, inp=[x], Tout=tf.float32
            )

            def mask_fn(d):
                x = d["image"]
                l = d["label"]

                return {"image": _fn(x), "label": l}

            ds = ds.map(mask_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds.prefetch(prefetch_size)

    if config.data.dataset in ["BRAIN"]:
        train_ds = DataLoader(
            train_ds, batch_size=config.training.batch_size, shuffle=False
        )
        eval_ds = DataLoader(
            eval_ds, batch_size=config.training.batch_size, shuffle=False
        )
        dataset_builder = None

        # return train_ds, eval_ds, None

    elif config.data.dataset in ["KNEE"]:
        train_ds = create_dataset(train_ds, train_split_name)
        eval_ds = create_dataset(eval_ds, eval_split_name)
    else:
        train_ds = create_dataset(dataset_builder, train_split_name)
        eval_ds = create_dataset(dataset_builder, eval_split_name, val=True)

    # #### Test if loader worked

    # for x in train_ds:
    #     print("Shape:", x["image"].shape)
    #     # print(x["image"].numpy().max())
    #     # q = np.quantile(x["image"].numpy(), 0.999)
    #     # plt.imshow(x["image"][0,0,128,], vmax=q)
    #     plot_slices(x["image"])
    #     plt.savefig(f"brain_sample.png")
    #     break
    # exit()

    return train_ds, eval_ds, dataset_builder
