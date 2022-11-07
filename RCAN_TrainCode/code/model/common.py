import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class PixelShuffle(nn.Module):

    def __init__(self, scale):
        super(PixelShuffle, self).__init__()
        self.scale=scale

    def forward(self, x):
        '''
        We use this complicated implementation since Hexagon AIP does not support 5D tensors (as warned by ``snpe-onnx-to-dlc``).

        If you only want to execute the model on CPU or GPU, 
        you just need to take care that Qualcomm Kryo CPUs and Ardeno GPUs only support 1D-5D transpose op.
        An alternative implementation would be:
        y=x
        B, iC, iH, iW = y.shape
        oC, oH, oW = iC//(self.scale*self.scale), iH*self.scale, iW*self.scale
        y = y.contiguous().view(torch.tensor((B*oC, self.scale, self.scale, iH, iW)).tolist())
        y = y.permute(0, 3, 1, 4, 2)
        y = y.contiguous().view(torch.tensor((B, oC, oH, oW)).tolist())
        return y
        '''
        y=x
        B, iC, iH, iW = y.shape
        oC, oH, oW = iC//(self.scale*self.scale), iH*self.scale, iW*self.scale
        y = torch.split(y, self.scale*self.scale, 1)
        y = [torch.split(sub, self.scale, 1) for sub in y]
        y = [[subsub.permute(0, 2, 3, 1).reshape(torch.tensor((B, iH, oW, 1)).tolist()) for subsub in sub] for sub in y]
        y = [torch.cat(sub,3) for sub in y]
        y = [sub.permute(0, 1, 3, 2).reshape(torch.tensor((B, 1, oH, oW)).tolist()) for sub in y]
        y = torch.cat(y, 1)
        return y

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, n_colors, bn=False, act=False, bias=True):

        m = []
        if scale == 4: # adapt to MobiSR
            m.append(conv(n_feat, 4*n_colors, 3, bias))
            m.append(PixelShuffle(2))
            if bn: m.append(nn.BatchNorm2d(n_colors))
            if act: m.append(act())
            m.append(conv(n_colors, 4*n_colors, 3, bias))
            m.append(PixelShuffle(2))
            if bn: m.append(nn.BatchNorm2d(n_colors))
            if act: m.append(act())
        elif scale == 2:
            m.append(conv(n_feat, 4*n_colors, 3, bias))
            m.append(PixelShuffle(2))
            if bn: m.append(nn.BatchNorm2d(n_colors))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Used by the candidate model m_rn
class rn_conv(nn.Module):
    def __init__(self, in_channels, kernel_size, r, bias=True):
        super(rn_conv, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//r, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//r, in_channels//r, kernel_size, bias=bias, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//r, in_channels, 1, bias=bias)
        )
    
    def forward(self, x):
        y = self.conv(x)
        y = y + x
        return y

## Used by the candidate model m_rxn
class rxn_conv(nn.Module):
    def __init__(self, in_channels, kernel_size, r, g, bias=True):
        super(rxn_conv, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//r, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//r, in_channels//r, kernel_size, bias=bias, padding=kernel_size//2, groups=g),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//r, in_channels, 1, bias=bias)
        )
    
    def forward(self, x):
        y = self.conv(x)
        y = y + x
        return y

## Used by the candidate model m_m1
class m1_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(m1_conv, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, bias=bias, padding=kernel_size//2, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

## Used by the candidate model m_m2
class m2_conv(nn.Module):
    def __init__(self, in_channels, kernel_size, e, bias=True):
        super(m2_conv, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * e, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * e, in_channels * e, kernel_size, padding=kernel_size//2, bias=bias, groups=in_channels * e),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * e, in_channels, 1, bias=bias),
        )

    def forward(self, x):
        y = self.conv(x)
        y = y + x
        return y


## Used by the model m_eff
class eff_conv(nn.Module):
    def __init__(self, in_channels, kernel_size, r=2, bias=True):
        super(eff_conv, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//r, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//r, in_channels//r, (1, kernel_size), bias=bias, padding=(0, kernel_size//2), groups=in_channels//r),
            nn.Conv2d(in_channels//r, in_channels//r, (kernel_size, 1), bias=bias, padding=(kernel_size//2, 0), groups=in_channels//r),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//r, in_channels, 1, bias=bias)
        )

    def forward(self, x):
        y = self.conv(x)
        y = y + x
        return y

## Optimized for mobile CPU/GPU/DSP
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        y = x
        B, C, H, W = y.shape
        y = y.permute(0, 2, 3, 1) # we do this as Adreno GPUS have a limitation on width*depth (i.e., shape[-1]*shape[-2] for a 4-dim tensor)
        shape = torch.tensor((B, H*W, self.groups, C//self.groups))
        y = y.contiguous().view(shape.tolist()) # we do not directly use y.view((B, H*W, self., C//self.)) because OnnxReshapeTranslation.extract_parameters does not support dynamic reshaping with a dynamically provided output shape
        y = y.permute(0, 1, 3, 2)
        shape = torch.tensor((B, H, W, C))
        y = y.contiguous().view(shape.tolist())
        y = y.permute(0, 3, 1, 2)
        return y

## Used by the model m_clc
class clc_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, g=16, bias=True):
        super(clc_conv, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, bias=bias, padding=kernel_size//2, groups=g),
            ChannelShuffle(g),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

## Used by the model m_s1
class s1_conv(nn.Module):
    def __init__(self, in_channels, kernel_size, r=2, g=4, bias=True):
        super(s1_conv, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//r, 1, bias=bias, groups=g),
            nn.ReLU(inplace=True),
            ChannelShuffle(g),
            nn.Conv2d(in_channels//r, in_channels//r, kernel_size, groups=in_channels//r, padding=kernel_size//2),
            nn.Conv2d(in_channels//r, in_channels, 1, groups=g)
        )
    
    def forward(self, x):
        y = self.conv(x)
        y = y+x
        return y

## Used by the model m_s2
class s2_conv(nn.Module):
    def __init__(self, in_channels, kernel_size, bias=True):
        super(s2_conv, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size -= 1
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size, bias=bias, padding=kernel_size//2),
            nn.Conv2d(in_channels//2, in_channels//2, 1, bias=bias)
        )
        self.shuffle = ChannelShuffle(2)
    
    def forward(self, x):
        y1, y2 = torch.split(x, self.in_channels//2, 1)
        y2 = self.conv(y2)
        y = torch.cat((y1, y2), 1)
        y = self.shuffle(y)
        return y


## add SELayer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## add SEResBlock
class SEResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(SEResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(SELayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x

        return res