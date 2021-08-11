import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter


class Normalize:
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {"image": img, "label": mask}


class UnNormalize:
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample.permute(0, 2, 3, 1)
        img = np.array(img).astype(np.float32)
        
        img *= self.std
        img += self.mean
        
        img[img < 0] = 0
        img[img > 1] = 1
        
        sample = img
        return sample
    

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample["image"]
        mask = sample["label"]
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {"image": img, "label": mask}


class RandomHorizontalFlip:
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img, "label": mask}


class RandomGaussianBlur:
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return {"image": img, "label": mask}


class RandomScaleCrop:
    def __init__(self, base_size, crop_size, fill=255, scale=(0.5, 2.0)):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale = scale
        self.fill = fill

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * self.scale[0]), int(self.base_size * self.scale[1]))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {"image": img, "label": mask}


class FixScale:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        return {"image": img, "label": mask}


class CenterCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size
        
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        
        width, height = img.size
        left = (width - self.crop_size) / 2
        top = (height - self.crop_size) / 2
        right = (width + self.crop_size) / 2
        bottom = (height + self.crop_size) / 2
        
        img = img.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))

        return {"image": img, "label": mask}


# CSRL Transform
class RandomHSV(object):
    """Generate randomly the image in hsv space."""
    def __init__(self, h_r, s_r, v_r):
        self.h_r = h_r
        self.s_r = s_r
        self.v_r = v_r

    def __call__(self, sample):
        image = sample['image']
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.int32)
        s = hsv[:, :, 1].astype(np.int32)
        v = hsv[:, :, 2].astype(np.int32)
        delta_h = np.random.randint(-self.h_r, self.h_r)
        delta_s = np.random.randint(-self.s_r, self.s_r)
        delta_v = np.random.randint(-self.v_r, self.v_r)
        h = (h + delta_h) % 180
        s = s + delta_s
        s[s > 255] = 255
        s[s < 0] = 0
        v = v + delta_v
        v[v > 255] = 255
        v[v < 0] = 0
        hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
        sample['image'] = image
        return sample


class RandomFlip(object):
    """Randomly flip image"""
    def __init__(self, threshold):
        self.flip_t = threshold

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['label']
        if np.random.rand() < self.flip_t:
            image_flip = np.flip(image, axis=1)
            segmentation_flip = np.flip(segmentation, axis=1)
            sample['image'] = image_flip
            sample['label'] = segmentation_flip
        return sample


class RandomScale(object):
    """Randomly scale image"""
    def __init__(self, scale_r, is_continuous=False):
        self.scale_r = scale_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['label']
        row, col, _ = image.shape
        rand_scale = np.random.rand() * (self.scale_r - 1 / self.scale_r) + 1 / self.scale_r
        img = cv2.resize(image, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
        seg = cv2.resize(segmentation, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
        sample['image'] = img
        sample['label'] = seg
        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        segmentation = segmentation[
            top: top + new_h,
            left: left + new_w
        ]
        sample['image'] = image
        sample['label'] = segmentation
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, is_continuous=False, fix=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
        self.fix = fix

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h, w):
            return sample
            
        if self.fix:
            h_rate = self.output_size[0] / h
            w_rate = self.output_size[1] / w
            min_rate = h_rate if h_rate < w_rate else w_rate
            new_h = h * min_rate
            new_w = w * min_rate
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        top = (self.output_size[0] - new_h) // 2
        bottom = self.output_size[0] - new_h - top
        left = (self.output_size[1] - new_w) // 2
        right = self.output_size[1] - new_w - left
        if self.fix:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        if 'label' in sample.keys():
            segmentation = sample['label']
            seg = cv2.resize(segmentation, dsize=(new_w, new_h), interpolation=self.seg_interpolation)
            if self.fix:
                seg = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            sample['label'] = seg
        sample['image'] = img
        return sample


class ToTensor2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        key_list = sample.keys()
        for key in key_list:
            if 'image' in key:
                image = sample[key]
                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                image = image.transpose((2, 0, 1))
                sample[key] = torch.from_numpy(image.astype(np.float32) / 255.0)
                # sample[key] = torch.from_numpy(image.astype(np.float32)/128.0-1.0)
            elif 'label' == key:
                segmentation = sample['label']
                sample['label'] = torch.from_numpy(segmentation.astype(np.float32))
            elif 'segmentation_onehot' == key:
                onehot = sample['segmentation_onehot'].transpose((2, 0, 1))
                sample['segmentation_onehot'] = torch.from_numpy(onehot.astype(np.float32))
            elif 'mask' == key:
                mask = sample['mask']
                sample['mask'] = torch.from_numpy(mask.astype(np.float32))
        return sample


class ToNumpy(object):
    """Convert IMAGE in sample to Numpy."""
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        return {"image": np.array(img), "label": np.array(mask)}
