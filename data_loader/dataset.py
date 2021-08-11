import pathlib
import numpy as np
import torch
import copy
from PIL import Image
from torchvision import transforms

from data_loader import custom_transforms as tr
from data_loader import PASCAL_DIR, PASCAL_NUM_CLASSES
from data_loader import CONTEXT_DIR, CONTEXT33_NUM_CLASSES, CONTEXT59_NUM_CLASSES

from base.base_dataset import BaseDataset, lbl_contains_unseen


class ContextSegmentation(BaseDataset):
    """
    Pascal Context dataset (Image: VOC2010(a part of 2012), Annotation:Context)
    """
    def __init__(
        self,
        unseen_classes_idx=[],
        seen_classes_idx=[],
        split="train",
        n_categories=33,
        transform=True,
        transform_args={},
        remv_unseen_img=True,
        ignore_bg=False,
        ignore_unseen=False,
    ):
        base_dir = pathlib.Path(CONTEXT_DIR)

        super().__init__(
            transform_args,
            base_dir,
            split,
            transform,
        )

        self.n_categories = n_categories
        self.remv_unseen_img = remv_unseen_img
        self.ignore_bg = ignore_bg
        self.ignore_unseen = ignore_unseen

        self._image_dir = pathlib.Path(PASCAL_DIR)
        
        if self.n_categories == 33:
            self.num_classes = CONTEXT33_NUM_CLASSES
        elif self.n_categories == 59:
            self.num_classes = CONTEXT59_NUM_CLASSES

        self._cat_dir = self._base_dir / ("%d_labels.pth" % (self.n_categories))  # Tensor dictionary
        self._cat = torch.load(self._cat_dir)

        self.unseen_classes_idx = unseen_classes_idx
        self.seen_classes_idx = seen_classes_idx

        self.im_ids = []
        self.categories = []

        lines = (self._base_dir / f"pascal_context_{self.split}.txt").read_text().splitlines()
        
        for ii, line in enumerate(lines):
            _image = self._image_dir / line.split()[0]
            _file_name = line.split()[1].split('/')[1].split('.')[0]  # file name
            assert _image.is_file(), _image
            assert _file_name in self._cat.keys(), _file_name

            # if unseen classes and training split
            if self.remv_unseen_img and (len(self.unseen_classes_idx) > 0) and (self.split == 'train'):
                cat = self._cat[_file_name].numpy()

                # Check rather label contains unseen classes index
                if lbl_contains_unseen(cat, self.unseen_classes_idx):
                    continue

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_file_name)

        assert len(self.images) == len(self.categories)

        # Display stats
        print("(context_%d) Number of images in %s: %d" % (self.n_categories, self.split, len(self.images)))

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split in ["train"]:
                sample = self.transform_tr(sample)
            elif self.split in ["val"]:
                sample = self.transform_val(sample)
        else:
            sample = self.transform_test(sample)

        sample["image_name"] = str(self.images[index])
        
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        target = self._cat[self.categories[index]].numpy()

        if self.ignore_bg:
            target[target == 0] = 255

        target_ = copy.deepcopy(target)
        if self.ignore_unseen:
            for ig_cls in self.unseen_classes_idx:
                target_[target == ig_cls] = 255
        _target = Image.fromarray(target_)
        
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.transform_args['base_size'],
                    crop_size=self.transform_args['crop_size'],
                    fill=255,
                    scale=(0.5, 2.0)
                ),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.FixScale(crop_size=self.transform_args['crop_size']),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def __str__(self):
        return f"CONTEXT(split={self.split})"


class VOCSegmentation(BaseDataset):
    """
    PascalVoc dataset
    """
    def __init__(
        self,
        unseen_classes_idx=[],
        seen_classes_idx=[],
        split="train",
        transform=True,
        transform_args={},
        remv_unseen_img=True,
        ignore_bg=False,
        ignore_unseen=False,
    ):
        base_dir = pathlib.Path(PASCAL_DIR)
        super().__init__(
            transform_args,
            base_dir,
            split,
            transform,
        )
        self.remv_unseen_img = remv_unseen_img
        self.ignore_bg = ignore_bg
        self.ignore_unseen = ignore_unseen

        self.num_classes = PASCAL_NUM_CLASSES
        if 'aug' not in self.split:
            self._image_dir = self._base_dir / "JPEGImages"
            self._cat_dir = self._base_dir / "SegmentationClass"
        else:
            self._image_dir = self._base_dir
            self._cat_dir = self._base_dir
    
        self.unseen_classes_idx = unseen_classes_idx
        self.seen_classes_idx = seen_classes_idx

        _splits_dir = self._base_dir / "ImageSets" / "Segmentation"

        self.im_ids = []
        self.categories = []

        lines = (_splits_dir / f"{self.split}.txt").read_text().splitlines()
        
        for ii, line in enumerate(lines):
            if 'aug' not in self.split:
                _image = self._image_dir / f"{line}.jpg"
                _cat = self._cat_dir / f"{line}.png"
            else:
                _image = self._image_dir / line.split()[0][1:]
                _cat = self._cat_dir / line.split()[1][1:]
            assert _image.is_file(), _image
            assert _cat.is_file(), _cat

            # Excluding training samples that contain unseen classes, only if it is training split
            if self.remv_unseen_img and len(self.unseen_classes_idx) > 0 and (self.split == "train_aug" or self.split == 'train'):
                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)

                # Check rather label contains unseen classes index
                if lbl_contains_unseen(cat, self.unseen_classes_idx):
                    continue

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split in ["trainval_aug", "trainval", "train_aug", "train"]:
                sample = self.transform_tr(sample)
            elif self.split in ["val_aug", "val"]:
                sample = self.transform_val(sample)
        else:
            sample = self.transform_test(sample)

        sample["image_name"] = str(self.images[index])
        
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        target = np.array(_target, dtype=np.uint8)

        # Ignore BG. classes for a SPNet setting
        if self.ignore_bg:
            target[target == 0] = 255

        # Ignore Unseen classes for a SPNet setting
        target_ = copy.deepcopy(target)
        if self.ignore_unseen:
            for ig_cls in self.unseen_classes_idx:
                target_[target == ig_cls] = 255
        _target = Image.fromarray(target_)
        
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.transform_args['base_size'],
                    crop_size=self.transform_args['crop_size'],
                    fill=255,
                    scale=(0.5, 1.5)
                ),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.FixScale(crop_size=self.transform_args['crop_size']),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def __str__(self):
        return f"VOC2012(split={self.split})"
