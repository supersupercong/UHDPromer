import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY

import torch.nn as nn
import torch

import torch.nn.functional as F
import numbers
from einops import rearrange

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

## AdaMean
def adaptive_mean_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size))
    return normalized_feat + style_mean.expand(size)

## AdaStd
def adaptive_std_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat) / content_std.expand(size)
    return normalized_feat * style_std.expand(size)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

##########################################################################
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim * ffn_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x = F.gelu(x)
#         x = self.project_out(x)
#         return x


##########################################################################
class Transformation_FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(Transformation_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, 2 * hidden_features, kernel_size=1, bias=bias)
        self.project_dwconv = nn.Conv2d(2 * hidden_features, 2 * hidden_features,
                                        kernel_size=3, padding=1, bias=bias, groups=2 * hidden_features)

        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, bias=bias,
                                        groups=hidden_features)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.project_y = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.fusion1 = nn.Conv2d(2*hidden_features, hidden_features, kernel_size=1, bias=bias)

        self.fusion2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)

    def forward(self, x, y=None):
        x1, x2 = self.project_dwconv(self.project_in(x)).chunk(2, dim=1)
        x1 = F.gelu(x1)

        if y is not None:
            y = self.project_y(y)
            fusion1 = self.fusion1(torch.cat([x2, y], dim=1))
            x1 = fusion1 * x1
            x2 = self.fusion2(fusion1)

        x1 = self.dwconv1(x1)
        x12 = x1 * x2
        x = self.project_out(x12)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, 2 * hidden_features, kernel_size=1, bias=bias)
        self.project_dwconv = nn.Conv2d(2 * hidden_features, 2 * hidden_features,
                                        kernel_size=3, padding=1, bias=bias, groups=2 * hidden_features)

        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, bias=bias,
                                        groups=hidden_features)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.project_y = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.fusion1 = nn.Conv2d(2*hidden_features, hidden_features, kernel_size=1, bias=bias)

        self.fusion2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)

    def forward(self, x, y=None):
        x1, x2 = self.project_dwconv(self.project_in(x)).chunk(2, dim=1)
        x1 = F.gelu(x1)

        x1 = self.dwconv1(x1)
        x12 = x1 * x2
        x = self.project_out(x12)
        return x

#############################################
class Fusion_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Fusion_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

    def intra_fusion_attention(self, q, y):
        b, c, h, w = q.size()

        k, v = self.kv(y).chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w).contiguous()

        return out

    def forward(self, x, y=None):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        if y is not None:
            q = self.intra_fusion_attention(q, y)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w).contiguous()

        out = self.project_out(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

    def intra_fusion_attention(self, q, y):
        b, c, h, w = q.size()

        k, v = self.kv(y).chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w).contiguous()

        return out

    def forward(self, x, y=None):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # if y is not None:
        #     q = self.intra_fusion_attention(q, y)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w).contiguous()

        out = self.project_out(out)
        return out


class LayerNorm_GRN(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
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
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None].contiguous() * x + self.bias[:, None, None].contiguous()
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Gated_Convolution(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, in_c, bias=False):
        super().__init__()
        dim = in_c
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LayerNorm_GRN(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(dim)
        self.pwconv2 = nn.Linear(dim, dim)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

        # x = input + self.drop_path(x)
        x = input + x
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat, scale):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=scale, stride=scale, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, scale):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * (scale*scale), kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.body(x)


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim=32, num_heads=1, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True,
                 LayerNorm_type='WithBias', attention_discrimination=True, ffn_discrimination=True, ffn_restormer=False):
        super(TransformerBlock, self).__init__()
        self.dim =dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if attention_discrimination is True:
            self.attn = Fusion_Attention(dim=dim,
                                  num_heads=num_heads,
                                  bias=bias)
        else:
            self.attn = Attention(dim=dim,
                                         num_heads=num_heads,
                                         bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_restormer = ffn_restormer
        if ffn_discrimination is True:
            self.ffn = Transformation_FeedForward(
                dim=dim,
                ffn_factor=ffn_expansion_factor,
                bias=bias)
        else:
            self.ffn = FeedForward(
                dim=dim,
                ffn_factor=ffn_expansion_factor,
                bias=bias)


        self.LayerNorm = LayerNorm(dim, LayerNorm_type)

    def forward(self, x, perception):
        # print('perception',perception.size())
        percetion = self.LayerNorm(perception)
        x = x + self.attn(self.norm1(x), percetion)
        x = x + self.ffn(self.norm2(x), percetion)
        return x


class ResBlock_TransformerBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """

    def __init__(self, dim=32, num_heads=1, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True,
                 LayerNorm_type='WithBias', attention_discrimination=True, ffn_discrimination=True, ffn_restormer=False, unit_num=3):
        super(ResBlock_TransformerBlock, self).__init__()
        self.unit_num = unit_num
        self.TransformerBlock = nn.ModuleList()
        self.prior_transformation = nn.ModuleList()

        for i in range(self.unit_num):
            self.prior_transformation.append(nn.Sequential(nn.Conv2d(3 * dim, dim, 1, bias=bias),
                                                         nn.Conv2d(dim, dim, 8, 8, bias=bias)))
            self.TransformerBlock.append(TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                match_factor=match_factor,
                ffn_expansion_factor=ffn_expansion_factor,
                scale_factor=scale_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                attention_discrimination=attention_discrimination,
                ffn_discrimination=ffn_discrimination,
                ffn_restormer=ffn_restormer))
    def get_masks(self, img_list):
        b, c, h, w = img_list[0].size()
        num_frames = len(img_list)

        img_list_copy = [img.detach() for img in img_list]  # detach backward
        # if self.is_mask_filter:  # mean filter
        #     img_list_copy = [calc_meanFilter(im, n_channel=3, kernel_size=5) for im in img_list_copy]

        delta = 1.
        mid_frame = img_list_copy[num_frames // 2]
        diff = torch.zeros_like(mid_frame)
        for i in range(num_frames):
            diff = diff + (img_list_copy[i] - mid_frame).pow(2)
        diff = diff / (2 * delta * delta)
        diff = torch.sqrt(torch.sum(diff, dim=1, keepdim=True))
        luckiness = torch.exp(-diff)  # (0,1)

        # print('luckiness', luckiness.size())

        luckiness = luckiness.expand(b, c, h, w)
        # print('luckiness = luckiness.expand(b, c, h, w)', luckiness.size())

        # sum_mask = torch.ones_like(flow_mask_list[0])
        # for i in range(num_frames): .expand(2,2,3)
        #     sum_mask = sum_mask * flow_mask_list[i]
        # sum_mask = torch.sum(sum_mask, dim=1, keepdim=True)
        # sum_mask = (sum_mask > 0).float()
        # luckiness = luckiness * sum_mask

        return luckiness

    def forward(self, input, perception):
        tmp = input
        for i in range(self.unit_num):
            #
            prior_transformation = self.prior_transformation[i](torch.cat(perception, dim=1))
            get_masks = self.get_masks([tmp, prior_transformation])
            tmp = self.TransformerBlock[i](tmp, get_masks)

        out = 0.2 * tmp + input
        return out


class Net(nn.Module):
    def __init__(self, channel_query_dict, number_block, num_heads=8, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True,
                 LayerNorm_type='WithBias', attention_discrimination=True, ffn_discrimination=True,SR_Guidance=True, ffn_restormer=False,unit_num=3):
        super().__init__()
        self.channel_query_dict = channel_query_dict
        self.enter = nn.Sequential(nn.Conv2d(3, channel_query_dict[256], 3, 1, 1))
        self.shallow = Gated_Convolution(channel_query_dict[256])
        self.middle = Gated_Convolution(channel_query_dict[256])
        self.deep = Gated_Convolution(channel_query_dict[256])
        # self.perception_fusion = Perception_fusion(channel_query_dict[256])
        self.block = nn.ModuleList()
        self.number_block = number_block
        for i in range(self.number_block):
            self.block.append(ResBlock_TransformerBlock(dim=channel_query_dict[256],
                                                        num_heads=num_heads,
                                                        match_factor=match_factor,
                                                        ffn_expansion_factor=ffn_expansion_factor,
                                                        scale_factor=scale_factor,
                                                        bias=bias,
                                                        LayerNorm_type=LayerNorm_type,
                                                        attention_discrimination=attention_discrimination,
                                                        ffn_discrimination=ffn_discrimination,
                                                        ffn_restormer=ffn_restormer,
                                                        unit_num=unit_num))
        self.downsample = Downsample(channel_query_dict[256], scale_factor)
        self.upsample = Upsample(channel_query_dict[256], scale_factor)

        self.SRNet_m1 = Gated_Convolution(channel_query_dict[256])
        self.SRNet_m2 = Gated_Convolution(channel_query_dict[256])
        self.SRNet_output = nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1)

        # self.fusion = nn.Conv2d(2*channel_query_dict[256], channel_query_dict[256], 1)

        self.out_m0 = Gated_Convolution(channel_query_dict[256])
        self.out_m1 = Gated_Convolution(channel_query_dict[256])
        self.out_m2 = Gated_Convolution(channel_query_dict[256])
        self.out = nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1)
        self.SR_Guidance = SR_Guidance
        if self.SR_Guidance is True:
            self.concat1 = nn.Conv2d(2 * channel_query_dict[256], channel_query_dict[256], 1)
            self.concat2 = nn.Conv2d(2 * channel_query_dict[256], channel_query_dict[256], 1)

    def forward(self, x):
        ori = x
        enter = self.enter(x)
        shallow = self.shallow(enter)
        middle = self.middle(shallow)
        deep = self.deep(middle)
        block = self.downsample(enter)
        block_input = block
        for i in range(self.number_block):
            block = self.block[i](block, [shallow, middle, deep])
        block = block_input + block
        upsample = self.upsample(block)

        SRNet_m1 = self.SRNet_m1(upsample)
        SRNet_m2 = self.SRNet_m2(SRNet_m1)
        SRNet_output = self.SRNet_output(SRNet_m2)

        out_m0 = self.out_m0(deep)
        if self.SR_Guidance is True:
            concat1 = self.concat1(torch.cat([SRNet_m1, out_m0], dim=1))
            out_m1 = self.out_m1(concat1)

            concat2 = self.concat2(torch.cat([SRNet_m2, out_m1], dim=1))
            out_m2 = self.out_m2(concat2)

            out = self.out(out_m2)
        else:
            out_m1 = self.out_m1(SRNet_m2)
            out_m2 = self.out_m2(out_m1)

            out = self.out(out_m2)
        return out + ori, SRNet_output + ori


@ARCH_REGISTRY.register()
class UHDPromer(nn.Module):
    def __init__(self,
                 *,
                 number_block,
                 num_heads=8,
                 match_factor=1,
                 ffn_expansion_factor=3,
                 scale_factor=8,
                 bias=True,
                 LayerNorm_type='WithBias',
                 attention_discrimination=True,
                 ffn_discrimination=True,
                 SR_Guidance=True,
                 ffn_restormer=False,
                 **ignore_kwargs):
        super().__init__()
        channel_query_dict = {256: 16}
        self.restoration_network = Net(channel_query_dict=channel_query_dict,
                                       number_block=number_block,
                                       num_heads=num_heads,
                                       match_factor=match_factor,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       scale_factor=scale_factor,
                                       bias=bias,
                                       LayerNorm_type=LayerNorm_type,
                                       attention_discrimination=attention_discrimination,
                                       ffn_discrimination=ffn_discrimination,
                                       SR_Guidance=SR_Guidance,
                                       ffn_restormer=ffn_restormer)

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def encode_and_decode(self, input, current_iter=None):

        out, SRNet_output = self.restoration_network(input)
        return out, SRNet_output

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    def check_image_size(self, x, window_size=16):
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % (window_size)) % (
            window_size)
        mod_pad_w = (window_size - w % (window_size)) % (
            window_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
        return x

    @torch.no_grad()
    def test(self, input):
        _, _, h_old, w_old = input.shape

        out, SRNet_output = self.encode_and_decode(input)

        return out, SRNet_output

    def forward(self, input):
        out, SRNet_output = self.encode_and_decode(input)

        return out, SRNet_output
