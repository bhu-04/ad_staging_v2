"""
Module 6 - DenseNet-121 Backbone (Algorithm 4)
==================================================
Implements the dense-connectivity principle behind DenseNet-121 -- every
layer within a block receives the concatenated feature maps of every
preceding layer in that block, then a 1x1 transition layer compresses
channels between blocks -- as a compact 2D network sized to train in
reasonable time on CPU rather than the full ~8M-parameter torchvision
DenseNet121 (timed at ~3.4s/iteration on this sandbox's CPU, which made a
12-epoch run take ~15 minutes for this module alone).

SCOPE NOTE (same substitution logic as cnn2d.py): 2D, not 3D, because the
dataset is 2D slices; compact rather than full-depth DenseNet-121 because
this increment runs on CPU with no GPU. The dense-block / transition-layer
/ global-pool / dropout-head structure is otherwise the genuine DenseNet
design (concatenated feature reuse across every layer in a block, channel
compression between blocks), not a renamed CNN2D. Growth rate, block
count, and layers-per-block are the parameters to scale up first if this
moves to GPU or the full torchvision densenet121 topology is wanted.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """BN -> ReLU -> Conv(3x3, growth_rate), output concatenated with input."""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, n_layers: int):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(n_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    """1x1 conv to halve channels, followed by 2x2 average pooling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))


class DenseNet2D(nn.Module):
    """Compact DenseNet standing in for Algorithm 4's DenseNet-121 (3D).
    Features a `self.features` attribute (dense blocks + transitions, pre
    global-pool) so visualize_pipeline.py's Grad-CAM hook works against it
    exactly like it does against CNN2D."""

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.35,
                 in_channels: int = 1, growth_rate: int = 8,
                 block_layers=(3, 3, 3)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        ch = 16
        blocks = []
        for i, n_layers in enumerate(block_layers):
            block = DenseBlock(ch, growth_rate, n_layers)
            blocks.append(block)
            ch = block.out_channels
            if i != len(block_layers) - 1:
                out_ch = ch // 2
                blocks.append(TransitionLayer(ch, out_ch))
                ch = out_ch
        blocks.append(nn.BatchNorm2d(ch))
        blocks.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*blocks)   # dense blocks + transitions + final BN/ReLU
        self.out_channels = ch

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)   # MC Dropout target layer
        self.classifier = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Optional transfer-learning backbone (Section 5 of the optimization pass)
# ---------------------------------------------------------------------------
class DenseNet2DPretrained(nn.Module):
    """torchvision densenet121, adapted for 1-channel input and 3-class
    output, with most of the backbone frozen and only the last dense block
    + classifier head trainable initially. Exposes `self.features` (the
    same attribute name CNN2D/DenseNet2D use) so visualize_pipeline.py's
    Grad-CAM hook works against it unmodified."""

    def __init__(self, base_model, num_classes: int = 3, dropout_p: float = 0.35,
                 freeze_until: str = "denseblock4"):
        super().__init__()
        # Adapt the stem conv for 1-channel grayscale input. Warm-start by
        # averaging the pretrained RGB kernel across the channel dim rather
        # than reinitializing from scratch, so the low-level edge/texture
        # filters aren't thrown away.
        old_conv = base_model.features.conv0
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                              stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        base_model.features.conv0 = new_conv

        self.features = base_model.features  # (B, 1024, H', W') pre-pool, matches DenseNet2D
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)  # MC-Dropout target layer
        in_features = base_model.classifier.in_features
        self.classifier = nn.Linear(in_features, num_classes)

        self._freeze_backbone(freeze_until)

    def _freeze_backbone(self, freeze_until: str) -> None:
        """Freeze every backbone param up to (not including) the named
        dense block, so only the last dense block + head train initially.
        Call `unfreeze_all()` afterwards for a fine-tuning pass.

        Also records which BatchNorm2d modules fall inside the frozen
        region (`self._frozen_bn_modules`). requires_grad=False on a BN
        layer's weight/bias only stops the *affine* params from learning;
        it does NOT stop running_mean/running_var from being updated by
        batch statistics whenever the module is in train() mode. Those are
        buffers, not parameters, and are governed purely by
        `module.training`. `train()` below uses this list to keep frozen
        BN layers in eval mode (so they use fixed running stats and never
        update them) independent of requires_grad -- see train() override.
        """
        unfreeze_from_here = False
        frozen_bn_modules = []
        for name, module in self.features.named_children():
            if name == freeze_until:
                unfreeze_from_here = True
            for p in module.parameters():
                p.requires_grad = unfreeze_from_here
            if not unfreeze_from_here:
                frozen_bn_modules.extend(
                    m for m in module.modules() if isinstance(m, nn.BatchNorm2d)
                )
        self._frozen_bn_modules = frozen_bn_modules

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        # Backbone is no longer conceptually "frozen" -- let all BatchNorm
        # layers (including these) follow model.train()/eval() normally
        # for a subsequent fine-tuning pass.
        self._frozen_bn_modules = []

    def train(self, mode: bool = True):
        """Same as nn.Module.train(), except frozen-backbone BatchNorm
        layers are always forced back into eval mode: their running
        statistics must not drift while their weights are frozen. This
        keeps the *smallest possible* footprint -- no other module's
        training/eval state, and no requires_grad flag, is touched -- and
        applies automatically to every model.train() call (training loop,
        retry attempts, etc.) without any change needed outside this class.
        """
        super().train(mode)
        if mode:
            for bn in getattr(self, "_frozen_bn_modules", []):
                bn.eval()
        return self

    def forward(self, x):
        f = torch.relu(self.features(x))
        f = self.global_pool(f).flatten(1)
        f = self.dropout(f)
        return self.classifier(f)


def build_densenet_backbone(num_classes: int = 3, dropout_p: float = 0.35,
                             pretrained: bool = None) -> nn.Module:
    """Factory used by demo mode (Section 5): try a pretrained torchvision
    DenseNet-121 adapted for grayscale/3-class input; if pretrained weights
    can't be fetched (no internet, no local cache) fall back to the
    compact from-scratch DenseNet2D above. Never raises for a missing
    network -- this must work fully offline."""
    from .. import config
    pretrained = pretrained if pretrained is not None else config.USE_PRETRAINED_DEMO_BACKBONE
    if not pretrained:
        return DenseNet2D(num_classes=num_classes, dropout_p=dropout_p)

    try:
        from torchvision.models import densenet121, DenseNet121_Weights
        base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model = DenseNet2DPretrained(base, num_classes=num_classes, dropout_p=dropout_p,
                                      freeze_until=config.FREEZE_BACKBONE_UNTIL)
        logging.getLogger(__name__).info(
            "Loaded pretrained torchvision DenseNet-121 (backbone frozen up to %s).",
            config.FREEZE_BACKBONE_UNTIL,
        )
        return model
    except Exception as e:  # no internet, no cached weights, etc.
        logging.getLogger(__name__).warning(
            "Pretrained DenseNet-121 unavailable (%s); falling back to the "
            "compact from-scratch DenseNet2D.", e,
        )
        return DenseNet2D(num_classes=num_classes, dropout_p=dropout_p)


if __name__ == "__main__":
    model = DenseNet2D(num_classes=3)
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}  (expected: (4, 3))")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")