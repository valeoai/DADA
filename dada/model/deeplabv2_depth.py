import torch
import torch.nn as nn

from advent.model.deeplabv2 import Bottleneck, ResNetMulti

AFFINE_PAR = True


class ResNetMultiDepth(ResNetMulti):
    def __init__(self, block, layers, num_classes, multi_level):
        super().__init__(block, layers, num_classes, multi_level)
        self.enc4_1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc4_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.dec4 = nn.Conv2d(128, 2048, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc4_1.weight.data.normal_(0, 0.01)
        self.enc4_2.weight.data.normal_(0, 0.01)
        self.enc4_3.weight.data.normal_(0, 0.01)
        self.dec4.weight.data.normal_(0, 0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.multi_level:
            seg_conv4 = self.layer5(x)  # produce segmap 1, not used in DADA
        else:
            seg_conv4 = None
        x4 = self.layer4(x)
        # encoder
        x4_enc = self.enc4_1(x4)
        x4_enc = self.relu(x4_enc)
        x4_enc = self.enc4_2(x4_enc)
        x4_enc = self.relu(x4_enc)
        x4_enc = self.enc4_3(x4_enc)
        depth = torch.mean(x4_enc, dim=1, keepdim=True)  # depth output
        # decoder
        x4_dec = self.dec4(x4_enc)
        x4_dec = self.relu(x4_dec)
        x4 = x4 * x4_dec  # feat-fusion
        seg_conv5 = self.layer6(x4)  # produce segmap 2
        return seg_conv4, seg_conv5, depth

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())    
        b.append(self.enc4_1.parameters())
        b.append(self.enc4_2.parameters())
        b.append(self.enc4_3.parameters())
        b.append(self.dec4.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


def get_deeplab_v2_depth(num_classes=16, multi_level=False):
    model = ResNetMultiDepth(
        Bottleneck, [3, 4, 23, 3], num_classes, multi_level
    )
    return model
