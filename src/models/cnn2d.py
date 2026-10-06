"""
Module 5 - Classification Backbone (CNN)
===========================================
Implements Algorithm 3 ("3D CNN") from the project document, adapted to a
2D convolutional backbone.

SCOPE NOTE: The project document specifies a *3D* CNN operating on
volumetric input. Because the actual Kaggle release provides individual 2D
axial slices rather than reconstructed 3D volumes (see data_acquisition.py),
this module implements the 2D analogue of Algorithm 3: the same
Conv -> BatchNorm -> ReLU -> Pool block structure, global average pooling,
and a dropout-regularized fully-connected head, but with nn.Conv2d in place
of nn.Conv3d. Re-stacking slices per subject into a volume and swapping in
Conv3d is a mechanical change once the full multi-slice-per-subject dataset
is available; the training/evaluation/uncertainty code around this model
does not depend on which one is used.

Dropout is placed deliberately (after global pooling, before the final
linear layer) so that `uncertainty/mc_dropout.py` can activate exactly this
layer at inference time per Algorithm 6, without touching BatchNorm.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CNN2D(nn.Module):
    """Compact 4-block 2D CNN backbone standing in for Algorithm 3's 3D CNN."""

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.35, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)  # <-- MC Dropout target layer
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        f = self.features(x)                 # (B, 128, H', W')
        f = self.global_pool(f).flatten(1)    # (B, 128)
        f = self.dropout(f)
        logits = self.classifier(f)           # (B, num_classes)
        return logits


if __name__ == "__main__":
    model = CNN2D(num_classes=3)
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    print(model)
    print(f"\nOutput shape: {tuple(out.shape)}  (expected: (4, 3))")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
