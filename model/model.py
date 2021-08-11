import torch
import torch.nn as nn

from torch.nn import functional as F
from model.deeplabv2 import DeepLabV2
from model.deeplabv3 import DeepLabV3Plus
from model.sync_batchnorm import SynchronizedBatchNorm2d


class GeneratorMLP(nn.Module):
    def __init__(
        self,
        embed_dim,
        feature_dim,
        hidden_size,
        pretrained=False,
        pretrained_path="",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout())
            return layers

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if hidden_size:
            self.model = nn.Sequential(
                *block(embed_dim, hidden_size),
                nn.Linear(hidden_size, feature_dim),
            )
        else:
            self.model = nn.Linear(embed_dim, feature_dim)

        self.model.apply(init_weights)

        if pretrained:
            self._load_pretrained_model(pretrained_path)
            print("Freeze Generator")
            self.set_parameter_requires_grad(self.model, freeze=True)

    def _load_pretrained_model(self, pretrained_path):
        pretrain_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        print("Get Pretrained Weight {%s}" % (pretrain_dict['arch']))
        self.load_state_dict(pretrain_dict['state_dict'], strict=False)
    
    def _set_parameter_requires_grad(model, freeze=False):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
    
    def forward(self, embd):
        features = self.model(embd)
        return features


class DeepLabV2_classifier_600(DeepLabV2):
    def __init__(
        self,
        sync_bn=True,
        num_classes=None,
        embed_dim=600,
        freeze_all_bn=False,
        freeze_backbone_bn=False,
        backbone_pretrained=False,
        imagenet_pretrained_path="",
        pretrained=False,
        pretrained_path="",
        freeze_backbone=False,
        freeze_aspp=False,
        freeze_pred_conv=False
    ):
        super(DeepLabV2_classifier_600, self).__init__(
            num_classes=embed_dim,  # Output dimension of ASPP is 600
            n_blocks=[3, 4, 23, 3],
            atrous_rates=[6, 12, 18, 24],
            sync_bn=sync_bn,
            freeze_backbone_bn=freeze_backbone_bn,
            pretrained=backbone_pretrained,
            pretrained_path=imagenet_pretrained_path,
        )

        self.pred_conv = nn.Conv2d(600, num_classes, kernel_size=1, stride=1)

        # Initialize the classifier
        nn.init.kaiming_normal_(self.pred_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.pred_conv.bias, 0)

        if pretrained:
            self._load_pretrained_model(pretrained_path)
            if freeze_backbone:
                print("Freeze backbone")
                self._set_parameter_requires_grad(self.backbone, freeze_backbone)
            if freeze_aspp:
                print("Freeze ASPP")
                self._set_parameter_requires_grad(self.aspp, freeze_aspp)
            if freeze_pred_conv:
                print("Freeze Classifier")
                self._set_parameter_requires_grad(self.pred_conv, freeze_pred_conv)

    def forward(self, x):
        feature = self.forward_before_class_prediction(x)
        logit = self.forward_class_prediction(feature)        
        return logit, feature

    def forward_before_class_prediction(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        out = F.relu(x)  # Add ReLU activation
        return out

    def forward_class_prediction(self, x):
        out = self.pred_conv(x)
        return out

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.pred_conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def _load_pretrained_model(self, pretrained_path):
        pretrain_dict = torch.load(pretrained_path)
        self.load_state_dict(pretrain_dict['state_dict'], strict=False)
