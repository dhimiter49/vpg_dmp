from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import densenet121
from torchvision.models import DenseNet121_Weights


"""
Three backbones for the value function. DenseNetUpsample is based on the original VPG
implementation. The other UNet methods are strongly based on this implementation
https://github.com/milesial/Pytorch-UNet. DenseNetUNet uses densent121 features for the
encoder and an adapted UNet decoder. For the VPG algorithm only DenseNetUpsample works
because of the rotation taking place during the VF forward pass.
"""
class DenseNetUpsample(nn.Module):
    def __init__(self, input_size, freeze, no_depth=False, bilinear=True):
        super(DenseNetUpsample, self).__init__()
        self.no_depth = no_depth
        self.feat = DenseNetFeatures(freeze).feat
        num_feat = 1024 if self.no_depth else 2048
        self.critic = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_feat)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv0', nn.Conv2d(num_feat, 64, kernel_size=1, stride=1, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))
        self.upsample = nn.Upsample(size=input_size, mode='bilinear')

    def features(self, rgb, depth):
        feat = self.feat(rgb)
        if not self.no_depth:
            depth = torch.cat([depth] * rgb.shape[1], dim=1)
            depth_feat = self.feat(depth)
            feat = torch.cat((feat, depth_feat), dim=1)
        return feat

    def forward(self, rgb, depth):
        feat = self.features(rgb, depth)
        return self.upsample(self.critic(feat))  # remove channel dim


class UNet(nn.Module):
    def __init__(self, n_channels, no_depth, freeze=False, bilinear=False):
        super(UNet, self).__init__()
        factor = 2 if bilinear else 1

        self.enc = Encoder(n_channels, factor, no_depth)
        dec_layers = []
        if not no_depth:
            dec_layers.append(Up(2048, 512 // factor, 2048, 1024, bilinear))
            dec_layers.append(Up(768, 256 // factor, 512, 256, bilinear))
            dec_layers.append(Up(384, 128 // factor, 256, 128, bilinear))
            dec_layers.append(Up(192, 64, 128, 64, bilinear))
            dec_layers.append(OutConv(64, 1))
        else:
            dec_layers.append(Up(1024, 512 // factor, bilinear=bilinear))
            dec_layers.append(Up(512, 256 // factor, bilinear=bilinear))
            dec_layers.append(Up(256, 128 // factor, bilinear=bilinear))
            dec_layers.append(Up(128, 64, bilinear=bilinear))
            dec_layers.append(OutConv(64, 1))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, rgb, depth):
        features = self.enc(rgb, depth)
        x = features[-1]
        for i, feat in enumerate(reversed(features[:-1])):
            x = self.dec[i](x, feat)
        return self.dec[-1](x).squeeze(1)


class DenseNetUNet(nn.Module):
    def __init__(self, freeze, no_depth, bilinear=False):
        super(DenseNetUNet, self).__init__()
        self.no_depth = no_depth
        self.enc = Encoder(None, None, no_depth, model=DenseNetFeatures(freeze).feat)
        dec_layers = []
        if not no_depth:
            if bilinear:
                dec_layers.append(Up(2560, 256, bilinear=bilinear))
                dec_layers.append(Up(512, 128, bilinear=bilinear))
                dec_layers.append(Up(256, 64, bilinear=bilinear))
                dec_layers.append(Up(192, 32, bilinear=bilinear))
                dec_layers.append(Up(36, 32, bilinear=bilinear))
                dec_layers.append(OutConv(32, 1))
            else:
                dec_layers.append(Up(1024, 256, 2048, 512, bilinear))
                dec_layers.append(Up(384, 128, 256, 128, bilinear))
                dec_layers.append(Up(192, 64, 128,  64, bilinear))
                dec_layers.append(Up(160, 32, 64, 32, bilinear))
                dec_layers.append(Up(36, 32, 32, 32,  bilinear))
                dec_layers.append(OutConv(32, 1))
        else:
            if bilinear:
                dec_layers.append(Up(1280, 256, bilinear=bilinear))
                dec_layers.append(Up(384, 128, bilinear=bilinear))
                dec_layers.append(Up(192, 64, bilinear=bilinear))
                dec_layers.append(Up(128, 32, bilinear=bilinear))
                dec_layers.append(Up(35, 32, bilinear=bilinear))
                dec_layers.append(OutConv(32, 1))
            else:
                dec_layers.append(Up(512, 256, 1024, 256, bilinear))
                dec_layers.append(Up(256, 128, 256, 128, bilinear))
                dec_layers.append(Up(128, 64, 128,  64, bilinear))
                dec_layers.append(Up(96, 32, 64, 32, bilinear))
                dec_layers.append(Up(35, 32, 32, 32,  bilinear))
                dec_layers.append(OutConv(32, 1))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, rgb, depth):
        features = self.enc(rgb, torch.cat([depth] * 3, dim=1))
        x = features[-1]
        for i, feat in enumerate(reversed(features[:-1])):
            x = self.dec[i](x, feat)
        input_ = rgb if self.no_depth else torch.cat([rgb, depth], dim=1)
        x = self.dec[-2](x, input_)
        return self.dec[-1](x)


class DoubleCriticWrapper(nn.Module):
    def __init__(self, freeze, no_depth, bilinear=False):
        super(DoubleCriticWrapper, self).__init__()
        self.no_depth = no_depth
        self.q1 = DenseNetUNet(freeze, no_depth, bilinear)
        self.q2 = copy.deepcopy(self.q1.dec)
        self.q2.apply(weights_init)

    def forward(self, rgb, depth):
        features = self.q1.enc(rgb, torch.cat([depth] * 3, dim=1))
        x, x_ = features[-1], features[-1]
        for i, feat in enumerate(reversed(features[:-1])):
            x = self.q1.dec[i](x, feat)
            x_ = self.q2[i](x_, feat)
        input_ = rgb if self.no_depth else torch.cat([rgb, depth], dim=1)
        x = self.q1.dec[-2](x, input_)
        x_ = self.q2[-2](x_, input_)
        return self.q1.dec[-1](x), self.q1.dec[-1](x_)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)


class Encoder(nn.Module):
    def __init__(self, n_channels, factor, no_depth, model=None):
        super(Encoder, self).__init__()
        self.no_depth = no_depth
        self.enc = nn.Sequential(*[
            DoubleConv(n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024 // factor),
        ]) if model is None else model
        if not self.no_depth:
            self.depth_enc = nn.Sequential(*[
                DoubleConv(1, 64),
                Down(64, 128),
                Down(128, 256),
                Down(256, 512),
                Down(512, 1024 // factor),
        ]) if model is None else model

    def forward(self, rgb, depth):
        x, feat = rgb, []
        for layer in self.enc:
            x = layer(x)
            feat.append(x)
        if not self.no_depth:
            x = depth
            for i, layer in enumerate(self.depth_enc):
                x = layer(x)
                feat[i] = torch.cat([feat[i], x], dim=1)
        return tuple(feat)


class DenseNetFeatures(nn.Module):
    def __init__(self, freeze):
        super(DenseNetFeatures, self).__init__()
        net = densenet121(DenseNet121_Weights.DEFAULT)
        if freeze:
            for p in net.features.parameters():
               p.requires_grad = False
        self.feat = nn.Sequential(*[
            net.features[:2],
            net.features[2:4],
            net.features[4:6],
            net.features[6:8],
            net.features[8:]
        ])

    def forward(self, x):
        return self.feat(x).flatten(1)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch, out_ch, in_ch_up=None, out_ch_up=None, bilinear=True):
        super().__init__()
        in_ch_up = in_ch if in_ch_up is None else in_ch_up
        out_ch_up = out_ch if out_ch_up is None else out_ch_up
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch_up, out_ch_up, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
