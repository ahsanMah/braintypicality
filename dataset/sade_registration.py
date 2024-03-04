import pdb

import ants
import generate_mri
from sade.configs.ve import biggan_config
from sade.datasets.loaders import get_image_files_list, get_val_transform
from tqdm import tqdm

CACHE_DIR = "/codespace/braintypicality/dataset/template_cache/"
config = biggan_config.get_config()
dataset_dir = config.data.dir_path
splits_dir = config.data.splits_dir

####
if not os.path.exists(procd_ref_img_path):
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

    procd_ref_img = ants.merge_channels((t1_ref_img * ref_img_mask, t2_ref_img * ref_img_mask))
    procd_ref_img_path = f"{CACHE_DIR}/cropped_niral_mni.nii.gz"
    procd_ref_img.to_filename(procd_ref_img_path)
####

savedir = f"/DATA/Users/amahmood/braintyp/spacing_{int(config.data.spacing_pix_dim)}"
os.makedirs(savedir, exist_ok=True)

img_loader = get_val_transform(config)
procd_ref_img_path = f"{CACHE_DIR}/cropped_niral_mni.nii.gz"
ref_img_tensor = img_loader({"image": procd_ref_img_path})["image"].numpy()
ref_img_post_transform = (ants.from_numpy(ref_img_tensor[0]) + 1) / 2

fnames = glob.glob(f"{dataset_dir}/*.nii.gz")

for fname in tqdm(fnames):
    img_tensor = img_loader({"image": fname})["image"].numpy()
    t1_img = (ants.from_numpy(img_tensor[0]) + 1) / 2

    sampleid = os.path.basename(fname).split(".nii.gz")[0]

    reg_dict = ants.registration(
        fixed=ref_img_post_transform, moving=t1_img, type_of_transform="SyN",
        outprefix=f"{savedir}/{sampleid}", write_composite_transform=True
    )