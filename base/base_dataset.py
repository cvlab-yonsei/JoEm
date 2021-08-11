import numpy as np

from torch.utils import data
from pathlib import Path


class BaseDataset(data.Dataset):
    def __init__(
        self,
        transform_args,
        base_dir,
        split,
        transform,
    ):
        super().__init__()
        self.transform_args = transform_args
        self._base_dir = Path(base_dir)
        self.split = split
        self.images = []
        self.transform = transform

    def __len__(self):
        return len(self.images)


def lbl_contains_unseen(lbl, unseen):
    unseen_pixel_mask = np.in1d(lbl.ravel(), unseen)
    if np.sum(unseen_pixel_mask) > 0:  # ignore images with any train_unseen pixels
        return True
    return False
