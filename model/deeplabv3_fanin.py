import math
import torch
import torch.nn as nn

from torch.nn import functional as F
from model.sync_batchnorm import SynchronizedBatchNorm2d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        output_stride,
        BatchNorm,
        pretrained=True,
        imagenet_pretrained_path="",
    ):
        self.inplanes = 64
        super().__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block,
            64,
            blocks=layers[0],
            stride=strides[0],
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            blocks=layers[1],
            stride=strides[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            blocks=layers[2],
            stride=strides[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.layer4 = self._make_MG_unit(
            block,
            512,
            blocks=blocks,
            stride=strides[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )
        self._init_weight()

        if pretrained:
            self._load_pretrained_model(imagenet_pretrained_path)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation, downsample, BatchNorm)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm)
            )

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=blocks[0] * dilation,
                downsample=downsample,
                BatchNorm=BatchNorm,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=stride,
                    dilation=blocks[i] * dilation,
                    BatchNorm=BatchNorm,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, imagenet_pretrained_path):
        """
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        """

        pretrain_dict = torch.load(imagenet_pretrained_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # Take pretrained weight except FC layers for ImageNet classification
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)

        self.load_state_dict(state_dict)


def ResNet101(output_stride, BatchNorm, pretrained=False, imagenet_pretrained_path=""):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        output_stride,
        BatchNorm,
        pretrained=pretrained,
        imagenet_pretrained_path=imagenet_pretrained_path,
    )
    return model


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, BatchNorm):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            BatchNorm(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

        
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, BatchNorm, global_avg_pool_bn=False):
        if global_avg_pool_bn:  # If Batchsize is 1, error occur.
            super(ASPPPooling, self).__init__(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                BatchNorm(out_channels),
                nn.ReLU())
        else:
            super(ASPPPooling, self).__init__(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)  # Pytorch official code: align_corners=False, ZS3: align_corners=True
    
    
class ASPP(nn.Module):
    def __init__(self, output_stride, BatchNorm, global_avg_pool_bn=False, in_channels=2048, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU()))
        
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            raise NotImplementedError
            
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, BatchNorm))

        modules.append(ASPPPooling(in_channels, out_channels, BatchNorm, global_avg_pool_bn))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
            
        self._init_weight()

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm, plus=True):
        super().__init__()

        self.plus = plus
        low_level_inplanes = 256
        self.project = nn.Sequential(
            nn.Conv2d(low_level_inplanes, 48, 1, bias=False),
            BatchNorm(48),
            nn.ReLU(inplace=True),
        )
        if self.plus:
            self.last_conv = nn.Sequential(
                nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        else:
            self.last_conv = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(),
            )

        self.pred_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self._init_weight()
    
    def forward(self, x, low_level_feat):
        feature = self.forward_before_class_prediction(x, low_level_feat)
        out = self.forward_class_prediction(feature)
        return out

    def forward_before_class_prediction(self, x, low_level_feat):
        if self.plus:
            low_level_feat = self.project(low_level_feat)
            x = F.interpolate(x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True)
            x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

    def forward_class_prediction(self, x):
        x = self.pred_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabV3(nn.Module):
    def __init__(
        self,
        sync_bn=True,
        output_stride=16,
        num_classes=21,
        freeze_all_bn=False,
        freeze_backbone_bn=False,
        global_avg_pool_bn=True,
        backbone_pretrained=True,
        imagenet_pretrained_path="",
        pretrained=False,
        pretrained_path="",
        freeze_backbone=False,
        freeze_aspp=False,
        freeze_decoder=False,
    ):
        super().__init__()

        if sync_bn:
            self.BatchNorm = SynchronizedBatchNorm2d
        else:
            self.BatchNorm = nn.BatchNorm2d

        self.backbone = ResNet101(output_stride,
                                  self.BatchNorm,
                                  pretrained=backbone_pretrained,
                                  imagenet_pretrained_path=imagenet_pretrained_path)
        self.aspp = ASPP(output_stride, self.BatchNorm, global_avg_pool_bn, in_channels=2048, out_channels=256)
        self.decoder = Decoder(num_classes, self.BatchNorm, False)

        # Freeze BNs
        self.freeze_all_bn = freeze_all_bn
        self.freeze_backbone_bn = freeze_backbone_bn
        self.freeze_bn()

        if pretrained:
            self._load_pretrained_model(pretrained_path)
            if freeze_backbone:
                print("Freeze backbone")
                self._set_parameter_requires_grad(self.backbone, freeze=freeze_backbone)
            if freeze_aspp:
                print("Freeze aspp")
                self._set_parameter_requires_grad(self.aspp, freeze=freeze_aspp)
            if freeze_decoder:
                print("Freeze decoder")
                self._set_parameter_requires_grad(self.decoder, freeze=freeze_decoder)

    def forward(self, input):
        feature = self.forward_before_class_prediction(input)
        out = self.forward_class_prediction(feature, input.shape[2:])
        return out, feature

    def forward_before_class_prediction(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder.forward_before_class_prediction(x, low_level_feat)
        return x

    def forward_class_prediction(self, x, input_size):
        x = self.decoder.forward_class_prediction(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return x

    def freeze_bn(self):
        if self.freeze_all_bn:
            print('FREEZE BN() - freeze all BNs')
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                    m.eval()
            
        elif self.freeze_backbone_bn:
            print('FREEZE BN() - freeze only backbone BNs')
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                    m.eval()
        else:
            print('FREEZE BN() - freeze nothing')

    def _set_parameter_requires_grad(model, freeze=False):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def _load_pretrained_model(self, pretrained_path):
        pretrain_dict = torch.load(pretrained_path)
        print("Get Pretrained Weight {%s}" % (pretrain_dict['arch']))
        print(self.load_state_dict(pretrain_dict['state_dict'], strict=False))


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

         
class DeepLabV3Plus(DeepLabV3):
    def __init__(
        self,
        sync_bn=True,
        output_stride=16,
        num_classes=21,
        freeze_all_bn=False,
        freeze_backbone_bn=False,
        global_avg_pool_bn=True,
        seperable_conv=False,
        backbone_pretrained=False,
        imagenet_pretrained_path="",
        pretrained=False,
        pretrained_path="",
        freeze_backbone=False,
        freeze_aspp=False,
        freeze_decoder=False,
    ):
        super().__init__(
            sync_bn,
            output_stride,
            num_classes,
            freeze_all_bn,
            freeze_backbone_bn,
            global_avg_pool_bn,
            backbone_pretrained,
            imagenet_pretrained_path,
        )

        self.decoder = Decoder(num_classes, self.BatchNorm, True)

        if seperable_conv:
            print("Using Seperable Conv")
            self.convert_to_separable_conv(self.aspp)
            self.convert_to_separable_conv(self.decoder)

        # if freeze_all_bn:
        #     self.freeze_all_bn = freeze_all_bn
        #     self.freeze_bn(backbone=False)

        # Re-call freeze_bn methods to freeze BN of new decoder.
        if freeze_all_bn:
            self.freeze_bn()

        if pretrained:
            # Load pretrained path
            self._load_pretrained_model(pretrained_path)
            if freeze_backbone:
                print("Freeze backbone")
                self._set_parameter_requires_grad(self.backbone, freeze=freeze_backbone)
            if freeze_aspp:
                print("Freeze aspp")
                self._set_parameter_requires_grad(self.aspp, freeze=freeze_aspp)
            if freeze_decoder:
                print("Freeze decoder")
                self._set_parameter_requires_grad(self.decoder, freeze=freeze_decoder)
        
    def convert_to_separable_conv(self, module):
        new_module = module
        if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
            new_module = AtrousSeparableConvolution(module.in_channels,
                                                    module.out_channels,
                                                    module.kernel_size,
                                                    module.stride,
                                                    module.padding,
                                                    module.dilation,
                                                    module.bias)
        for name, child in module.named_children():
            new_module.add_module(name, self.convert_to_separable_conv(child))
        return new_module
