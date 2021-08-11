import numpy as np
import os.path as osp

PASCAL_DIR = "/dataset/PASCALVOC/VOCdevkit/VOC2012"

CONTEXT_DIR = "/dataset/CONTEXT/"

DATASETS_IMG_DIRS = {"pascal": PASCAL_DIR}

VOC = ['background',
       'airplane', 'bicycle', 'bird', 'boat', 'bottle',
       'bus', 'car', 'cat', 'chair', 'cow',
       'diningtable', 'dog', 'horse', 'motorbike', 'person',
       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
       
PASCAL_NUM_CLASSES = len(VOC)

CONTEXT33 = ['background'] + [
    label.decode().replace(" ", "")
    for idx, label in np.genfromtxt(
        osp.join(CONTEXT_DIR, '33_labels.txt'),
        delimiter=":",
        dtype=None,
    )
]
CONTEXT33_NUM_CLASSES = len(CONTEXT33)

CONTEXT59 = ['background'] + [
    label.decode().replace(" ", "")
    for idx, label in np.genfromtxt(
        osp.join(CONTEXT_DIR, '59_labels.txt'),
        delimiter=":",
        dtype=None,
    )
]
CONTEXT59_NUM_CLASSES = len(CONTEXT59)


def get_unseen_idx(n_unseen_classes, dataset='pascal'):
    if dataset.lower() == 'pascal':
        if n_unseen_classes == 2:
            return_list = sorted([10, 14])
        elif n_unseen_classes == 4:
            return_list = sorted([10, 14, 1, 18])
        elif n_unseen_classes == 5:  # SPNet setting
            return_list = sorted([16, 17, 18, 19, 20])
        elif n_unseen_classes == 6:
            return_list = sorted([10, 14, 1, 18, 8, 20])
        elif n_unseen_classes == 8:
            return_list = sorted([10, 14, 1, 18, 8, 20, 19, 5])
        elif n_unseen_classes == 10:
            return_list = sorted([10, 14, 1, 18, 8, 20, 19, 5, 9, 16])

        elif n_unseen_classes == 1:
            return_list = sorted([20])
        else:
            return_list = []
    
    elif dataset.lower() == 'context33' or dataset.lower() == 'context59':
        if n_unseen_classes == 2:
            return_list = sorted([10, 14])  # cow, mbike
        elif n_unseen_classes == 4:
            return_list = sorted([10, 14, 18, 8])  # + sofa, cat
        elif n_unseen_classes == 6:
            return_list = sorted([10, 14, 18, 8, 4, 32])  # + boat, fence
        elif n_unseen_classes == 8:
            return_list = sorted([10, 14, 18, 8, 4, 32, 3, 20])  # bird, tv
        elif n_unseen_classes == 10:
            return_list = sorted([10, 14, 18, 8, 4, 32, 3, 20, 38, 2])  # keyboard, aeroplane
        else:
            return_list = []

    else:
        print('Error!')
        return_list = []

    return sorted(return_list)


def get_seen_idx(n_unseen_classes, ignore_bg=False, dataset='pascal'):
    if dataset.lower() == 'pascal':
        n_class = PASCAL_NUM_CLASSES
    elif dataset.lower() == 'context33':
        n_class = CONTEXT33_NUM_CLASSES
    elif dataset.lower() == 'context59':
        n_class = CONTEXT59_NUM_CLASSES
    else:
        print('Error!')
        return []

    seen_classes_idx = [i for i in range(n_class)]
    for idx in get_unseen_idx(n_unseen_classes, dataset):
        seen_classes_idx.remove(idx)
    if ignore_bg:
        seen_classes_idx.remove(0)
    return sorted(seen_classes_idx)
