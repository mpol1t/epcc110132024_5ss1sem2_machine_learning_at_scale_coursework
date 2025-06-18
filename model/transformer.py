from collections import OrderedDict
from functools import partial
from typing import Type, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from ruamel.yaml.scalarfloat import ScalarFloat
from torch.nn.modules.module import Module

from model.utils import DropPath, trunc_normal_
from utils.y_params import YParams


class MLPBlock(nn.Module):
    def __init__(
            self,
            prefix: str,
            index: int,
            in_features: int,
            out_features: int,
            drop_rate: float,
            act_layer: Type[nn.Module]
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            OrderedDict(
                [
                        (f'{prefix}_lin_{index}', nn.Linear(in_features, out_features)),
                        (f'{prefix}_act_{index}', act_layer()),
                        (f'{prefix}_drop_{index}', nn.Dropout(drop_rate))
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLP(nn.Module):
    def __init__(
            self,
            layers: List[int],
            drop_rate: float,
            act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            OrderedDict(
                [
                        (
                                f'mlp_{i}',
                                MLPBlock(
                                    prefix='mlp',
                                    index=i,
                                    in_features=in_features,
                                    out_features=out_features,
                                    drop_rate=drop_rate,
                                    act_layer=act_layer
                                )
                        ) for i, (in_features, out_features) in enumerate(zip(layers, layers[1:]))
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: ScalarFloat = 0.,
            proj_drop: ScalarFloat = 0.,
            norm_layer: partial = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Attention mechanism, implementing scaled dot product attention.

        :param x: Input tensor with dimensions (batch_size, num_patches, dim).
        :return: Output tensor after attention and projection.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop_rate: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm
    ) -> None:
        super().__init__()

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop_rate,
            norm_layer=norm_layer
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(
            layers=[dim, mlp_hidden_dim, dim],
            drop_rate=drop_rate,
            act_layer=act_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a single transformer block, combining attention, MLP, and residual connections.

        :param x: Input tensor of shape (batch_size, sequence_length, dim).
        :return: Output tensor after processing through the transformer block.
        """
        y = self.attn(self.norm1(x))

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size: Tuple[int, int] = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768
    ) -> None:
        super().__init__()

        # grid of patches
        h, w = img_size
        self.h = h // patch_size
        self.w = w // patch_size

        num_patches = self.h * self.w

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embeds an image into a sequence of patches suitable for transformer processing.

        :param x: Input image tensor.
        :return: Sequence of flattened embedded patches.
        """
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
            self,
            depth: int,
            qkv_bias: bool,
            img_size: Tuple[int, int],
            in_chans: int,
            out_chans: int,
            embed_dim: int,
            num_heads: int,
            mlp_ratio: float,
            drop_rate: float,
            patch_size: int,
            attn_drop_rate: float,
            drop_path_rate: float,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            **kwargs
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_ch = out_chans
        self.drop_rate = drop_rate

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_rate=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer
                    ) for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.out_size = self.out_ch * self.patch_size * self.patch_size

        self.head = nn.Linear(embed_dim, self.out_size, bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        """
        Initializes weights of the Vision Transformer model using truncated normal for linear layers and constant
        initialization for normalization layers.

        :param m: Module to initialize.
        """

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepares the token embeddings by applying patch embeddings and adding positional encodings to the patches.

        :param x: Input tensor representing the batch of images.
        :return: Tensor with positional encodings added to patch embeddings.
        """
        x = self.patch_embed(x)  # patch linear embedding
        x = x + self.pos_embed  # add positional encoding to each token

        return self.pos_drop(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the final processing head to the output of the transformer to generate the final image output.

        :param x: Input tensor from the final transformer block.
        :return: Final output image tensor reshaped to the original dimensions.
        """
        B, _, _ = x.shape  # B x N x embed_dim
        x = x.reshape(B, self.patch_embed.h, self.patch_embed.w, self.embed_dim)
        B, h, w, _ = x.shape

        # apply head
        x = self.head(x)
        x = x.reshape(shape=(B, h, w, self.patch_size, self.patch_size, self.out_ch))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_ch, self.img_size[0], self.img_size[1]))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the complete forward pass of the Vision Transformer model.

        :param x: Input tensor representing the batch of images.
        :return: The final output of the model after processing through the transformer blocks and output head.
        """
        x = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.forward_head(x)

        return x


def transformer(params: YParams, **kwargs) -> VisionTransformer:
    """
    Factory function to create a VisionTransformer model with specific configurations.

    :param params: Configuration parameters containing model settings.
    :param kwargs: Additional keyword arguments for fine-tuning.
    :return: An instance of VisionTransformer configured according to provided parameters.
    """
    return VisionTransformer(
        depth=params.depth,
        qkv_bias=params.qkv_bias,
        img_size=params.img_size,
        in_chans=params.n_in_channels,
        out_chans=params.n_out_channels,
        drop_rate=params.dropout,
        patch_size=params.patch_size,
        embed_dim=params.embed_dim,
        num_heads=params.num_heads,
        mlp_ratio=params.mlp_ratio,
        drop_path_rate=params.dropout,
        attn_drop_rate=params.dropout,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
