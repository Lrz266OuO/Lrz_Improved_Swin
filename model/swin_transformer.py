from torch import meshgrid
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, window_size):

def window_reverse(windows, window_size, H, W):

class Mlp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        return x

class WindowAttention(nn.Moudle):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

    def extra_repr(self) -> str:

    def flops(self, N):


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resulotion.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    # TODO: 修改dropout参数，0.3

    def __init__(self):
        super().__init__()

    def forward(self, x):

    def extra_repr(self) -> str:

    def flops(self):


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

        patches分辨率减半,通道数加倍

        Args:
            input_resolution (tuple[int]): Resolution of input feature.
            dim (int): Number of input channels.
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):

    def extra_repr(self) -> str:

    def flops(self):


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resolution.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):

    def extra_repr(self) -> str:

    def flops(self):


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

        Args:
            img_size (int): Image size.  Default: 224.
            patch_size (int): Patch token size. Default: 4.
            in_chans (int): Number of input image channels. Default: 3.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):

    def flops(self):


class SwinTransformer(nn.Module):
    r""" Swin Transformer
            A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
              https://arxiv.org/pdf/2103.14030

        Args:
            img_size (int | tuple(int)): Input image size. Default 224
            patch_size (int | tuple(int)): Patch size. Default: 4
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            embed_dim (int): Patch embedding dimension. Default: 96
            depths (tuple(int)): Depth of each Swin Transformer layer.
            depth 指的是每一个stage中block的数量[2,2,6,2]
            num_heads (tuple(int)): Number of attention heads in different layers.
            window_size (int): Window size. Default: 7
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
            drop_rate (float): Dropout rate. Default: 0
            attn_drop_rate (float): Attention dropout rate. Default: 0
            drop_path_rate (float): Stochastic depth rate. Default: 0.1
            norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
            ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
            patch_norm (bool): If True, add normalization after patch embedding. Default: True
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self):
        super().__init__()


