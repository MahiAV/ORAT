"""Dataset loaders for the vision experiments.

We follow the same protocol as ``Sequential_rank_1.ipynb``:

* MNIST and CIFAR-10 use the first 5 classes as the "old" task and the last
  5 classes as the "new" task we adapt to.
* CIFAR-100 uses classes 0..49 as old and 50..99 as new.

Only the *new* classes are used to evaluate adaptation quality (the test set
in :func:`make_dataloaders` returns the new-class subset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms


DATASET_REGISTRY = {
    "mnist": {
        "input_size": 1 * 28 * 28,
        "num_classes": 10,
        "old_classes": list(range(5)),
        "new_classes": list(range(5, 10)),
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
        "torchvision_cls": torchvision.datasets.MNIST,
    },
    "cifar10": {
        "input_size": 3 * 32 * 32,
        "num_classes": 10,
        "old_classes": list(range(5)),
        "new_classes": list(range(5, 10)),
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        ),
        "torchvision_cls": torchvision.datasets.CIFAR10,
    },
    "cifar100": {
        "input_size": 3 * 32 * 32,
        "num_classes": 100,
        "old_classes": list(range(50)),
        "new_classes": list(range(50, 100)),
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        ),
        "torchvision_cls": torchvision.datasets.CIFAR100,
    },
}


@dataclass
class VisionLoaders:
    name: str
    input_size: int
    num_classes: int
    old_classes: List[int]
    new_classes: List[int]
    train_old: DataLoader
    train_new: DataLoader
    test_old: DataLoader
    test_new: DataLoader


def _class_subset(dataset, class_indices: List[int]) -> Subset:
    """Return the subset of ``dataset`` whose targets fall in ``class_indices``."""
    targets = torch.as_tensor(dataset.targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for c in class_indices:
        mask |= targets == c
    indices = torch.where(mask)[0].tolist()
    return Subset(dataset, indices)


def make_dataloaders(
    dataset: str,
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> VisionLoaders:
    """Build train/test dataloaders for the old- and new-class splits."""
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose one of {list(DATASET_REGISTRY)}.")

    cfg = DATASET_REGISTRY[dataset]
    cls = cfg["torchvision_cls"]
    transform = cfg["transform"]

    train_full = cls(data_dir, train=True, download=True, transform=transform)
    test_full = cls(data_dir, train=False, download=True, transform=transform)

    train_old = _class_subset(train_full, cfg["old_classes"])
    train_new = _class_subset(train_full, cfg["new_classes"])
    test_old = _class_subset(test_full, cfg["old_classes"])
    test_new = _class_subset(test_full, cfg["new_classes"])

    def loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return VisionLoaders(
        name=dataset,
        input_size=cfg["input_size"],
        num_classes=cfg["num_classes"],
        old_classes=list(cfg["old_classes"]),
        new_classes=list(cfg["new_classes"]),
        train_old=loader(train_old, shuffle=True),
        train_new=loader(train_new, shuffle=True),
        test_old=loader(test_old, shuffle=False),
        test_new=loader(test_new, shuffle=False),
    )
