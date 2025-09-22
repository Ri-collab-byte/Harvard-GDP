# utils/image_datasets.py (supervised-only loader for RNFL/GRAPE npz)
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset

class Longitudinal_Dataset(Dataset):
    def __init__(self, data_path, subset="train", outcome_type="progression_outcome_td_pointwise_no_p_cut",
                 modality=1, resolution=224, data_type="label", need_shift=False, stretch=False):
        self.data_path = os.path.join(data_path, subset)
        self.files = sorted([f for f in os.listdir(self.data_path) if f.endswith(".npz")])
        self.outcome_type = outcome_type
        self.modality = modality
        self.resolution = resolution
        self.data_type = data_type

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, self.files[idx])
        dat = np.load(path, allow_pickle=True)
        img = dat["image"].astype(np.float32)
        # normalize
        img = (img - img.mean()) / (img.std() + 1e-6)
        img = torch.tensor(img, dtype=torch.float32)

        # fetch label
        meta = json.loads(dat["meta"].item()) if isinstance(dat["meta"].item(), str) else dat["meta"].item()
        y = None
        for key in [self.outcome_type, "progression.td_pointwise_no_p_cut", "progression", "label"]:
            if key in meta:
                try:
                    y = float(meta[key])
                except:
                    pass
                break
        if y is None:
            y = -1.0
        y = torch.tensor(y, dtype=torch.float32)

        return img, y, path

