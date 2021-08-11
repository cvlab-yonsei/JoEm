# Exploiting a Joint Embedding Space for Generalized Zero-Shot Semantic Segmentation

This is an official implementation of the paper "Exploiting a Joint Embedding Space for Generalized Zero-Shot Semantic Segmentation", accepted to ICCV2021.

For more information, please checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/JoEm/)] and the paper [[arXiv](https://arxiv.org/pdf/)].

## Pre-requisites
This repository uses the following libraries:
* Python (3.6)
* Pytorch (1.8.1)

## Getting Started

### Datasets

#### VOC
The structure of data path should be organized as follows:
```bash
/dataset/PASCALVOC/VOCdevkit/VOC2012/                         % Pascal VOC datasets root
/dataset/PASCALVOC/VOCdevkit/VOC2012/JPEGImages/              % Pascal VOC images
/dataset/PASCALVOC/VOCdevkit/VOC2012/SegmentationClass/       % Pascal VOC segmentation maps
/dataset/PASCALVOC/VOCdevkit/VOC2012/ImageSets/Segmentation/  % Pascal VOC splits
```


#### CONTEXT
The structure of data path should be organized as follows:
```bash
/dataset/context/                                 % Pascal CONTEXT dataset root
/dataset/context/59_labels.pth                    % Pascal CONTEXT segmentation maps
/dataset/context/pascal_context_train.txt         % Pascal CONTEXT splits
/dataset/context/pascal_context_val.txt           % Pascal CONTEXT splits
/dataset/PASCALVOC/VOCdevkit/VOC2012/JPEGImages/  % Pascal VOC images
```

### Training
We use DeepLabV3+ with ResNet-101 as our visual encoder. Following [ZS3Net](https://github.com/valeo/ZS3), ResNet-101 is initialized with the pre-trained weights for ImageNet classification, where training samples of seen classes are used only. ([weights here](https://github.com/))

#### VOC
```Shell
python train_pascal_zs3setting.py -c configs/config_pascal_zs3setting.json -d 0,1,2,3
```

* Trained visual and semantic encoder weights
    -  [2 unseen classes](https://github.com/)
    -  [4 unseen classes](https://github.com/)
    -  [6 unseen classes](https://github.com/)
    -  [8 unseen classes](https://github.com/)
    -  [10 unseen classes](https://github.com/)


#### CONTEXT
```Shell
python train_context_zs3setting.py -c configs/config_context_zs3setting.json -d 0,1,2,3
```

* Trained visual and semantic encoder weights
    -  [2 unseen classes](https://github.com/)
    -  [4 unseen classes](https://github.com/)
    -  [6 unseen classes](https://github.com/)
    -  [8 unseen classes](https://github.com/)
    -  [10 unseen classes](https://github.com/)

### Testing

#### VOC
```Shell
python train_pascal_zs3setting.py -c configs/config_pascal_zs3setting.json -d 0,1,2,3 -r <visual encoder>.pth --test
```
#### CONTEXT
```Shell
python train_pascal_zs3setting.py -c configs/config_pascal_zs3setting.json -d 0,1,2,3 -r <visual encoder>.pth --test
```

## Acknowledgements
* This template is borrowed from [pytorch-template](https://github.com/victoresque/pytorch-template).