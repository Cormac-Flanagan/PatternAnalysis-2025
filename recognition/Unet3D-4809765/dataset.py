import numpy as np
from utils import load_data_3D
from torch.utils.data import Dataset
from pathlib import Path
import torch
import kornia.augmentation as K

class NiiPairDataset(Dataset):
    def __init__(
        self,
        root_dir,
        preload=True,
        early_stop=0,
        transform: K.container.AugmentationSequential | None = None,
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
        if early_stop > 0:
            self.samples = self.samples[:early_stop]

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
            )
            print("Preloading LFOV images...")
            self.lfov_data = load_data_3D(
                lfov_files,
                normImage=True,
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
