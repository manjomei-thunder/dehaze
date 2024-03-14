import argparse


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import yaml

import Res2Net as Pre_Res2Net
import os
import common
import numpy as np
from timm.models.layers import trunc_normal_, DropPath

from config import get_config
from models import build_model


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        # self.stage1 = self.stages[0]
        # self.stage2 = self.stages[1]
        # self.stage3 = self.stages[2]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, layer):
        # for i in range(3): #4
            x = self.downsample_layers[layer](x)
            x = self.stages[layer](x)
        # return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
            return x
    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class priorKnow(nn.Module):
    def __init__(self):
        super(priorKnow, self).__init__()
        imagenet_model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        # print(imagenet_model)

        checkpoint = torch.load('convnext_small_1k_224_ema.pth')
        # print(torch.load(checkpoint))
        print(checkpoint['model'])
        imagenet_model.load_state_dict(checkpoint['model'] )
        self.encoder = imagenet_model
        self.att1 = CALayer(384)
        self.att2 = CALayer(192)
        self.att3 = CALayer(96)
        self.att4 = CALayer(24)
        # self.att5 = CALayer(6)

        # self.conv2 = nn.Conv2d()
        self.upscale1 = sub_pixel(2)
        self.conv2 = nn.Conv2d(6 ,12, 1,1)
        self.conv4 = nn.Conv2d(96,192,1,1)
        self.conv3 = nn.Conv2d(48,96,1,1)
        self.conv5 = nn.Conv2d(3,6,1,1)
        self.conv6 = nn.Conv2d(6, 32, 1, 1)
        self.upscale2 = sub_pixel(4)
        self.rel = nn.ReLU()
    def forward(self, x):
        # x = self.encoder.stage1(x)   #96,H,W
        #
        # x = self.encoder.stage2(x)   #192,H,W
        # x = self.encoder.stage3(x)   #384,H,W
        x0 = self.encoder.forward_features(x, 0)       #96, 64, 64
        x1 = self.encoder.forward_features(x0, 1)       #192, 32, 32
        x2 = self.encoder.forward_features(x1, 2)     #384, 16, 16
        x2 = self.att1(x2)
        x2 = self.upscale1(x2) #96, 32,32
        x2 = self.conv4(x2) #192, 32,32
        x2 = x2 + x1
        x2 = self.att2(x2)
        x2 = self.upscale1(x2) #48, 64, 64
        x2 = self.conv3(x2)  # 96, 64, 64
        x2 = x2 + x0  # 96, 64, 64

        # x2 = x + x2
        x2 = self.upscale2(x2) # 6, 256, 256
        x = self.conv5(x)

        x2 = x + x2
        print(x2.shape )

        # x2 = self.att5(x2)
        # x2 = self.upscale1(x2) #6, 128, 128
        # x2 = self.conv2(x2)
        # x2 = self.upscale1(x2)
        x2 = self.conv6(x2)
        x2 = self.rel(x2)
        # print(  x2.shape,x1.shape)
        # x2 = self.att(x2)
        # x2 = self.upscale1(x2)

        return x2


class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x
## Channel Attention (CA) Layer
class RC_CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(RC_CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(RC_CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res = res+ x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        # print(x.shape, res.shape)
        res = res+ x
        return res


class rcan(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(rcan, self).__init__()
        n_resgroups = 5
        n_resblocks = 10
        n_feats = 32
        kernel_size = 3
        reduction = 8
        act = nn.ReLU(True)

        # self.conv1 = nn.Conv2d(3, n_feats, kernel_size=3, padding=1, bias=True)
        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            conv(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x_feat = self.head(x)
        res = self.body(x_feat)
        out_feat = res + x_feat
        return out_feat


###################################################################################################################
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x = self.layer3(x_layer2)  # x16

        return x, x_layer1, x_layer2


######################
# decoder
######################
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res


class Enhancer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enhancer, self).__init__()

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.tanh = nn.Tanh()

        self.refine1 = nn.Conv2d(in_channels, 20, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)

        self.refine3 = nn.Conv2d(20 + 4, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()

        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze


class Dehaze(nn.Module):
    def __init__(self, imagenet_model):
        super(Dehaze, self).__init__()

        self.encoder = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101 = Pre_Res2Net.Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101.load_state_dict(torch.load(os.path.join(imagenet_model,'res2net101_v1b_26w_4s-0812c246.pth')))
        pretrained_dict = res2net101.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)                # encoder=Res2Net has pretrained weight

        self.mid_conv = DehazeBlock(default_conv, 1024, 3)

        self.up_block1 = nn.PixelShuffle(2)
        self.attention1 = DehazeBlock(default_conv, 256, 3)
        self.attention2 = DehazeBlock(default_conv, 192, 3)
        self.enhancer = Enhancer(28, 28)


    def forward(self, input):
        x, x_layer1, x_layer2 = self.encoder(input)     # [8 1024 16 16], [8 256 64 64], [8 512 32 32] = [8 3 256 256]

        x_mid = self.mid_conv(x)

        x = self.up_block1(x_mid)                       # [8 1024 16 16] -> [8 256 32 32]
        x = self.attention1(x)                          

        x = torch.cat((x, x_layer2), 1)                 # [8, 768, 32, 32] = 256+512
        x = self.up_block1(x)                           # [8, 192, 64, 64]
        x = self.attention2(x)

        x = torch.cat((x, x_layer1), 1)                 # [8, 448, 64, 64] = 192+256
        x = self.up_block1(x)                           # [8, 112, 128, 128]
        x = self.up_block1(x)                           # [8, 28, 256, 256]

        dout2 = self.enhancer(x)
        #torch.Size([2, 28, 256, 256])

        return dout2



class DehazeSwinT(nn.Module):
    def __init__(self, imagenet_model):
        super(DehazeSwinT, self).__init__()
        
        checkpoint = torch.load('swinv2_base_patch4_window8_256.pth', map_location='cpu')
        imagenet_model.load_state_dict(checkpoint['model'])
        self.encoder = imagenet_model

        # Incorporate with SwinTransformerV2
        self.mid_conv = DehazeBlock(default_conv, 1024, 3)

        self.up_block1 = nn.PixelShuffle(2)
        self.attention1 = DehazeBlock(default_conv, 256, 3)
        self.attention2 = DehazeBlock(default_conv, 192, 3)
        self.attention3 = DehazeBlock(default_conv, 112, 3)
        self.enhancer = Enhancer(15, 15)


    def forward(self, input):
        x, layer_feature = self.encoder(input)      # [0-2]: [4096,128] [1024,256] [256,512]  [3]=x: [64,1024]

        # change dimension
        x, feature1, feature2, feature3 = x.transpose(1,2), layer_feature[0].transpose(1,2), layer_feature[1].transpose(1,2), layer_feature[2].transpose(1,2)
        x = torch.reshape(x, (x.shape[0], x.shape[1], int(np.sqrt(x.shape[2])), int(np.sqrt(x.shape[2]))))
        feature1 = torch.reshape(feature1, (feature1.shape[0], feature1.shape[1], int(np.sqrt(feature1.shape[2])), int(np.sqrt(feature1.shape[2]))))
        feature2 = torch.reshape(feature2, (feature2.shape[0], feature2.shape[1], int(np.sqrt(feature2.shape[2])), int(np.sqrt(feature2.shape[2]))))
        feature3 = torch.reshape(feature3, (feature3.shape[0], feature3.shape[1], int(np.sqrt(feature3.shape[2])), int(np.sqrt(feature3.shape[2]))))

        x_mid = self.mid_conv(x)                    # [8, 1024, 8, 8] = [8, 1024, 8, 8]

        x = self.up_block1(x_mid)                   # [8, 256, 16, 16] = [8, 1024, 8, 8]
        x = self.attention1(x)

        x = torch.cat((x, feature3), 1)             # [8, 768, 16, 16]
        x = self.up_block1(x)                       # [8, 192, 32, 32]
        x = self.attention2(x)

        x = torch.cat((x, feature2), 1)             # [8, 448, 32, 32] = 192+256
        x = self.up_block1(x)                       # [8, 112, 64, 64]
        x = self.attention3(x)

        x = torch.cat((x, feature1), 1)             # [8, 240, 64, 64] = 112+128
        x = self.up_block1(x)
        x = self.up_block1(x)

        dout2 = self.enhancer(x)

        return dout2



class fusion_refine(nn.Module):
    def __init__(self, imagenet_model, rcan_model):
        super(fusion_refine, self).__init__()

        # first branch
        if imagenet_model.__class__.__name__ == 'SwinTransformerV2':
            self.feature_extract=DehazeSwinT(imagenet_model)    # parameters: 109313725
        else:
            self.feature_extract=Dehaze(imagenet_model)  # parameters: 49347811
            
        # second branch
        self.pre_trained_rcan= rcan()                     # parameters: 996651
        self.pre_trained_convnext = priorKnow()
        # tail
        if imagenet_model.__class__.__name__ == 'SwinTransformerV2':
            self.tail1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(47, 3, kernel_size=7, padding=0), nn.Tanh())
        else:
            self.tail1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(60, 3, kernel_size=7, padding=0), nn.Tanh())

        

    def forward(self, input):

        feature=self.feature_extract(input)                     # input:[8,3,256,256]   -> feature:[8,28,256,256]
        # rcan_out = self.pre_trained_rcan(input)
        conv_out = self.pre_trained_convnext(input)

        x = torch.cat([feature, conv_out], 1)                   # swin 15+32=47
        # x1 = torch.cat([feature, rcan_out], 1)                   # swin 15+32=47

        feat_hazy = self.tail1(x)

        return feat_hazy
#         return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)  # ,

            # nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
if __name__ == '__main__':
    input = torch.rand(1, 3, 256, 256).cuda()
    parser = argparse.ArgumentParser(description='RCAN-Dehaze-teacher')
    parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
    parser.add_argument('-train_batch_size', help='Set the training batch size', default=20, type=int)
    parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--model_save_dir', type=str, default='./output_img/output_result_fromHuan_val4450_intrain2625')
    parser.add_argument('--log_dir', type=str, default=None)
    # --- Parse hyper-parameters test --- #
    parser.add_argument('--test_dataset', type=str, default='')
    parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
    parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1,  type=int)
    parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
    parser.add_argument('--imagenet_model', default='', type=str, help='load trained model or not')
    parser.add_argument('--rcan_model', default='', type=str, help='load trained model or not')
    parser.add_argument('--ckpt_path', default='', type=str, help='path to model to be loaded')
    parser.add_argument('--hazy_data', default='', type=str, help='apply on test data or val data')
    parser.add_argument('--cropping', default='6', type=int, help='crop the 4k*6k image to # of patches for testing')

    # --- SwinTransformer Parameter --- #
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )         # required
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    #parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel') # required

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')           # --fused_window_process
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')




    args = parser.parse_args()
    config = get_config(args)
    swv2_model = build_model(config)
    swi = fusion_refine( swv2_model,args.rcan_model).cuda()
    swi.feature_extract.load_state_dict(torch.load('swinv2_base_patch4_window8_256.pth')['model'])
    # swi = priorKnow().cuda()
    output = swi(input)
    print(output.shape)