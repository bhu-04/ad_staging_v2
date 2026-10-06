"""
Module 7 - Vision Transformer Backbone (Algorithm 5)
=====================================================

Compact 2D Vision Transformer for the 4-class Alzheimer's staging task.

Adaptation:
- Input: grayscale 2D MRI slice
- Patch embedding via strided Conv2d
- Learnable CLS token
- Learnable positional embeddings
- Transformer encoder
- Classification head

The positional embedding is interpolated when the runtime image size
produces a different number of patches from the model's configured
reference image size. This keeps the model compatible with both demo
and full image-resolution profiles.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config


class PatchEmbedding(nn.Module):
    """
    Non-overlapping patch extraction + linear projection.

    Implemented as a strided convolution.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(
                f"Image size {img_size} must be divisible by "
                f"patch size {patch_size}."
            )

        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)

        # (B, embed_dim, H/P, W/P)
        # -> (B, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        return x


class ViT2D(nn.Module):
    """
    Compact Vision Transformer for grayscale MRI slices.

    Supports runtime image sizes that differ from the reference
    positional-embedding size through 2D positional interpolation.
    """

    def __init__(
        self,
        num_classes: int = 4,
        img_size: int | None = None,
        patch_size: int | None = None,
        embed_dim: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
        mlp_dim: int | None = None,
        dropout_p: float | None = None,
        in_channels: int = 1,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Resolve configuration lazily.
        # ---------------------------------------------------------

        img_size = (
            img_size
            if img_size is not None
            else config.IMAGE_SIZE[0]
        )

        patch_size = (
            patch_size
            if patch_size is not None
            else config.VIT_PATCH_SIZE
        )

        embed_dim = (
            embed_dim
            if embed_dim is not None
            else config.VIT_EMBED_DIM
        )

        depth = (
            depth
            if depth is not None
            else config.VIT_DEPTH
        )

        num_heads = (
            num_heads
            if num_heads is not None
            else config.VIT_NUM_HEADS
        )

        mlp_dim = (
            mlp_dim
            if mlp_dim is not None
            else config.VIT_MLP_DIM
        )

        dropout_p = (
            dropout_p
            if dropout_p is not None
            else config.DROPOUT_P
        )

        # ---------------------------------------------------------
        # Patch embedding.
        # ---------------------------------------------------------

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.patch_size = patch_size
        self.reference_grid_size = (
            self.patch_embed.grid_size
        )

        n_patches = self.patch_embed.n_patches

        # ---------------------------------------------------------
        # CLS token.
        # ---------------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim,
            )
        )

        # ---------------------------------------------------------
        # Learnable positional embedding.
        #
        # Shape:
        #   [1, 1 + patches, embed_dim]
        #
        # First position = CLS token.
        # Remaining positions = spatial patch positions.
        # ---------------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                n_patches + 1,
                embed_dim,
            )
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        # ---------------------------------------------------------
        # Transformer encoder.
        # ---------------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        # ---------------------------------------------------------
        # Classification head.
        # ---------------------------------------------------------

        self.norm = nn.LayerNorm(embed_dim)

        # MC Dropout target.
        self.dropout = nn.Dropout(
            p=dropout_p
        )

        self.head = nn.Linear(
            embed_dim,
            num_classes,
        )

    # -----------------------------------------------------------------
    # Positional embedding interpolation
    # -----------------------------------------------------------------

    def _get_positional_embedding(
        self,
        num_patches: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Return positional embeddings matching the current number
        of image patches.

        If the runtime patch grid equals the reference grid, use the
        learned embedding directly.

        Otherwise interpolate the spatial patch embeddings using
        bicubic interpolation.

        The CLS positional embedding is kept unchanged.
        """

        # Current stored positional embedding:
        #
        # [CLS] + spatial patch positions
        #
        stored_pos = self.pos_embed

        stored_num_patches = (
            stored_pos.shape[1] - 1
        )

        # Fast path.
        if stored_num_patches == num_patches:

            return stored_pos.to(
                device=device,
                dtype=dtype,
            )

        # ---------------------------------------------------------
        # Determine square spatial grids.
        # ---------------------------------------------------------

        old_grid = int(
            math.sqrt(stored_num_patches)
        )

        new_grid = int(
            math.sqrt(num_patches)
        )

        if old_grid * old_grid != stored_num_patches:
            raise RuntimeError(
                "Stored ViT positional embedding does not represent "
                f"a square patch grid: {stored_num_patches} patches."
            )

        if new_grid * new_grid != num_patches:
            raise RuntimeError(
                "Runtime ViT patch count does not represent "
                f"a square patch grid: {num_patches} patches."
            )

        # ---------------------------------------------------------
        # Separate CLS and spatial embeddings.
        # ---------------------------------------------------------

        cls_pos = stored_pos[:, :1, :]

        patch_pos = stored_pos[:, 1:, :]

        # ---------------------------------------------------------
        # Convert:
        #
        # [1, old_grid², embed_dim]
        #
        # into:
        #
        # [1, embed_dim, old_grid, old_grid]
        # ---------------------------------------------------------

        patch_pos = patch_pos.reshape(
            1,
            old_grid,
            old_grid,
            -1,
        )

        patch_pos = patch_pos.permute(
            0,
            3,
            1,
            2,
        )

        # ---------------------------------------------------------
        # Interpolate spatial positional embeddings.
        # ---------------------------------------------------------

        patch_pos = F.interpolate(
            patch_pos,
            size=(
                new_grid,
                new_grid,
            ),
            mode="bicubic",
            align_corners=False,
        )

        # ---------------------------------------------------------
        # Convert back:
        #
        # [1, embed_dim, new_grid, new_grid]
        #
        # -> [1, new_grid², embed_dim]
        # ---------------------------------------------------------

        patch_pos = patch_pos.permute(
            0,
            2,
            3,
            1,
        )

        patch_pos = patch_pos.reshape(
            1,
            num_patches,
            -1,
        )

        # ---------------------------------------------------------
        # Reattach CLS positional embedding.
        # ---------------------------------------------------------

        pos = torch.cat(
            [
                cls_pos,
                patch_pos,
            ],
            dim=1,
        )

        return pos.to(
            device=device,
            dtype=dtype,
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, x):

        B = x.shape[0]

        # ---------------------------------------------------------
        # Patch extraction + embedding.
        # ---------------------------------------------------------

        x = self.patch_embed(x)

        # Shape:
        # (B, n_patches, embed_dim)

        num_patches = x.shape[1]

        # ---------------------------------------------------------
        # CLS token.
        # ---------------------------------------------------------

        cls = self.cls_token.expand(
            B,
            -1,
            -1,
        )

        # ---------------------------------------------------------
        # Concatenate CLS + patches.
        # ---------------------------------------------------------

        x = torch.cat(
            [
                cls,
                x,
            ],
            dim=1,
        )

        # ---------------------------------------------------------
        # Runtime-compatible positional embedding.
        # ---------------------------------------------------------

        pos = self._get_positional_embedding(
            num_patches=num_patches,
            device=x.device,
            dtype=x.dtype,
        )

        x = x + pos

        # ---------------------------------------------------------
        # Transformer encoder.
        # ---------------------------------------------------------

        x = self.encoder(x)

        # ---------------------------------------------------------
        # CLS representation.
        # ---------------------------------------------------------

        x = self.norm(
            x[:, 0]
        )

        # ---------------------------------------------------------
        # Dropout.
        # ---------------------------------------------------------

        x = self.dropout(x)

        # ---------------------------------------------------------
        # Classification.
        # ---------------------------------------------------------

        return self.head(x)


# ---------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    model = ViT2D(
        num_classes=4,
        img_size=96,
    )

    dummy = torch.randn(
        4,
        1,
        96,
        96,
    )

    out = model(dummy)

    print(
        f"Output shape: {tuple(out.shape)} "
        f"(expected: (4, 4))"
    )

    print(
        f"Total parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )