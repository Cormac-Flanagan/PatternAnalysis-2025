from typing import Union
import numpy as np
import nibabel as nib
from tqdm import tqdm, utils
from torch.utils.data import Dataset
from pathlib import Path
import torch
import kornia.augmentation as K

EARLY = 1

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1
    return res


def load_data_3D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype: Union[np.dtype, type] = np.float32,
    getAffines=False,
    orient=False,
    early_stop=False,
):
    """
    Load medical image data from names , cases list provided into a list for each.
    This function pre - allocates 5 D arrays for conv3d to avoid excessive memory usage.
    normImage : bool ( normalise the image 0.0 -1.0)
    orient : Apply orientation and resample image ? Good for images with large slice &
    thickness or anisotropic resolution
    dtype : Type of the data . If dtype = np . uint8 , it is assumed that the data is &
    labels
    early_stop : Stop loading pre - maturely ? Leaves arrays mostly empty , for quick &
    loading and testing scripts .
    """
    affines = []

    interp = "linear"
    if dtype == np.uint8:
        interp = "nearest"

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])

    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]

    if categorical:
        first_case = to_channels(first_case)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]
        inImage = inImage[:, :, :depth]
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)

            images[
                i,
                : inImage.shape[0],
                : inImage.shape[1],
                : inImage.shape[2],
                : inImage.shape[3],
            ] = inImage
        else:
            images[
                i,
                : inImage.shape[0],
                : inImage.shape[1],
                : inImage.shape[2],
            ] = inImage

        affines.append(affine)
        if i > EARLY and early_stop:
            break
    if getAffines:
        return images, affines
    else:
        return images


class NiiPairDataset(Dataset):
    def __init__(
        self,
        root_dir,
        preload=True,
        early_stop=False,
        transform: K.container.AugmentationSequential |None=None,
    ):
        """
        root_dir/
          ├── semantics_labels_only/*.nii.gz
          └── semantics_MRs/*.nii.gz
        """
        self.transforms = transform
        self.root_dir = Path(root_dir)
        self.semantic_dir = self.root_dir / "semantic_labels_only"
        self.lfov_dir = self.root_dir / "semantic_MRs"

        self.preload = preload

        # Match SEMANTIC and LFOV pairs by basename
        self.samples = []
        for semantic_path in self.semantic_dir.glob("*.nii.gz"):
            base_name = semantic_path.stem
            if base_name.endswith(".nii"):
                base_name = base_name[:-12]
            lfov_path = self.lfov_dir / f"{base_name}LFOV.nii.gz"
            if lfov_path.exists():
                self.samples.append((semantic_path, lfov_path))
            else:
                print(f"Missing file for {base_name}")

        self.random_crop = K.RandomCrop3D((128, 128, 128), same_on_batch=True)
        # Optionally preload everything in bulk
        if preload:
            semantic_files = [str(s[0]) for s in self.samples]
            lfov_files = [str(s[1]) for s in self.samples]

            print("Preloading SEMANTIC images...")
            self.semantic_data = load_data_3D(
                semantic_files,
                categorical=True,
                dtype=np.uint8,
                early_stop=early_stop,
            )
            print("Preloading LFOV images...")
            self.lfov_data = load_data_3D(
                lfov_files,
                normImage=True,
                early_stop=early_stop,
            )
        else:
            self.semantic_data = None
            self.lfov_data = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            semantic_np = self.semantic_data[idx]
            lfov_np = self.lfov_data[idx]
            lfov_np = (lfov_np - lfov_np.min()) / (lfov_np.max() - lfov_np.min() + 1e-8)
            semantic = (
                torch.tensor(semantic_np).permute(3, 0, 1, 2).float().unsqueeze(0)
            )
            lfov = torch.from_numpy(lfov_np).float().unsqueeze(0)

            if self.transforms is not None:
                lfov, semantic = self.transforms(lfov, semantic)

            lfov = self.random_crop(lfov)
            semantic = self.random_crop.forward(
                semantic, params=self.random_crop._params
            ).long()

            return lfov.squeeze(0), semantic.squeeze(0)
        else:
            semantic_path, lfov_path = self.samples[idx]
            lfov_np, lfov_aff = load_data_3D(
                [str(lfov_path)],
                normImage=True,
                categorical=False,
            )
            semantic_np, semantic_aff = load_data_3D(
                [str(semantic_path)],
                normImage=False,
                categorical=True,
                dtype=np.uint8,
            )

            lfov_tensor = torch.from_numpy(lfov_np[0]).unsqueeze(0)
            semantic_tensor = (
                torch.from_numpy(semantic_np[0]).permute(3, 0, 1, 2).float()
            )

            return lfov_tensor, semantic_tensor
