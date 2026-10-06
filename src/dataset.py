"""
Volumetric Data Loader (Data Management Layer)
================================================
Wraps the manifest + preprocessing + augmentation modules into a standard
PyTorch Dataset/DataLoader pair, matching the "Volumetric Data Loader"
component named in the project's architecture table (here loading 2D
standardized slices rather than 3D volumes, per the dataset-format note in
data_acquisition.py).

PERF NOTES (optimization pass):
  - __getitem__ now calls preprocessing.get_cached_slice() instead of
    preprocessing.preprocess_slice(), so the JPEG-decode/resize/normalize
    work happens at most once per file per process instead of once per
    file per epoch per model.
  - DataLoader worker/pin_memory/persistent_workers settings are now
    centralized in config.py instead of hardcoded num_workers=0 here.
  - Class weights are computed straight from the manifest (pandas/numpy),
    not by iterating the DataLoader -- see train.class_weights_from_manifest.
  - Class balancing now applies EITHER the sampler OR the loss weighting
    (config.CLASS_BALANCE_STRATEGY), never both -- see config.py's docstring
    for why stacking them was overcompensating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from . import config, preprocessing


class ADStagingDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, filepaths: list, transform=None):
        self.transform = transform
        subset = manifest[manifest["filepath"].isin(filepaths)].reset_index(drop=True)
        # Preserve filepath order as given (already shuffled upstream by the splitter)
        order = {p: i for i, p in enumerate(filepaths)}
        subset["_order"] = subset["filepath"].map(order)
        self.manifest = subset.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        std_arr = preprocessing.get_cached_slice(row["filepath"])
        if self.transform:
            x = self.transform(std_arr)
        else:
            x = torch.from_numpy(std_arr).unsqueeze(0).float()
        y = int(row["class_idx"])
        return x, y


def make_balanced_sampler(labels: list) -> WeightedRandomSampler:
    """Per-sample inverse-frequency weights for WeightedRandomSampler, so
    the TRAIN loader draws roughly equal numbers of CN/MCI/AD per epoch.
    Only used when config.CLASS_BALANCE_STRATEGY == 'sampler'."""
    labels = np.asarray(labels)
    class_counts = np.bincount(labels, minlength=len(config.CLASS_NAMES))
    class_counts = np.clip(class_counts, 1, None)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(manifest: pd.DataFrame, split_result: dict,
                       train_transform, eval_transform,
                       batch_size: int = None,
                       balance_train: bool = None) -> dict:
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    # balance_train=None -> defer to the single config switch. Passing an
    # explicit True/False still overrides it for callers that need to.
    use_sampler = (config.CLASS_BALANCE_STRATEGY == "sampler") if balance_train is None else balance_train

    splits = split_result["splits"]
    datasets = {
        "train": ADStagingDataset(manifest, splits["train"], transform=train_transform),
        "val": ADStagingDataset(manifest, splits["val"], transform=eval_transform),
        "test": ADStagingDataset(manifest, splits["test"], transform=eval_transform),
    }

    loader_kwargs = dict(
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
    )

    if use_sampler:
        train_labels = datasets["train"].manifest["class_idx"].tolist()
        train_sampler = make_balanced_sampler(train_labels)
        train_loader = DataLoader(datasets["train"], batch_size=batch_size,
                                   sampler=train_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(datasets["train"], batch_size=batch_size,
                                   shuffle=True, **loader_kwargs)

    loaders = {
        "train": train_loader,
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, **loader_kwargs),
    }
    return loaders


if __name__ == "__main__":
    from . import data_acquisition, splitting, augmentation

    manifest = data_acquisition.build_manifest()
    split_result = splitting.subject_stratified_split(manifest)
    loaders = build_dataloaders(
        manifest, split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )
    for name, loader in loaders.items():
        x, y = next(iter(loader))
        print(f"{name}: {len(loader.dataset)} samples, batch x={tuple(x.shape)}, y={tuple(y.shape)}")
