from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _limit_dataset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=limit, replace=False)
    return Subset(dataset, indices.tolist())


def _dataset_class(dataset_name: str):
    name = dataset_name.lower()
    if name == "cifar10":
        return datasets.CIFAR10
    if name == "cifar100":
        return datasets.CIFAR100
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _dataset_root(data_dir: Path, dataset_name: str) -> Path:
    name = dataset_name.lower()
    if data_dir.name.lower() == name:
        return data_dir
    return data_dir / name


def _cifar_stats(dataset_name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if dataset_name.lower() == "cifar100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def _imagenet_stats() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_image_datasets(
    dataset_name: str,
    data_dir: Path,
    train_limit: int | None,
    test_limit: int | None,
    seed: int,
) -> tuple[Dataset, Dataset, list[str]]:
    dataset_cls = _dataset_class(dataset_name)
    root = _dataset_root(data_dir, dataset_name)
    transform = transforms.Compose([transforms.ToTensor()])
    train_base = dataset_cls(root=str(root), train=True, download=True, transform=transform)
    test_base = dataset_cls(root=str(root), train=False, download=True, transform=transform)
    return (
        _limit_dataset(train_base, train_limit, seed),
        _limit_dataset(test_base, test_limit, seed + 1),
        list(train_base.classes),
    )


def _deep_transforms(
    dataset_name: str,
    model_name: str,
    image_size: int | None,
) -> tuple[transforms.Compose, transforms.Compose]:
    if model_name == "convnext_tiny":
        size = image_size or 224
        mean, std = _imagenet_stats()
        train_steps = [
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        test_steps = [
            transforms.Resize((size, size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        return transforms.Compose(train_steps), transforms.Compose(test_steps)

    size = image_size or 32
    mean, std = _cifar_stats(dataset_name)
    if size == 32:
        train_crop = transforms.RandomCrop(32, padding=4)
        test_resize = None
    else:
        train_crop = transforms.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True)
        test_resize = transforms.Resize((size, size), antialias=True)

    train_steps = [
        train_crop,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    test_steps = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if test_resize is not None:
        test_steps.insert(0, test_resize)
    return transforms.Compose(train_steps), transforms.Compose(test_steps)


def get_tensor_datasets(
    dataset_name: str,
    data_dir: Path,
    train_limit: int | None,
    test_limit: int | None,
    seed: int,
    model_name: str,
    image_size: int | None = None,
) -> tuple[Dataset, Dataset, list[str]]:
    dataset_cls = _dataset_class(dataset_name)
    root = _dataset_root(data_dir, dataset_name)
    train_transform, test_transform = _deep_transforms(dataset_name, model_name, image_size)
    train_base = dataset_cls(
        root=str(root),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_base = dataset_cls(
        root=str(root),
        train=False,
        download=True,
        transform=test_transform,
    )
    return (
        _limit_dataset(train_base, train_limit, seed),
        _limit_dataset(test_base, test_limit, seed + 1),
        list(train_base.classes),
    )


def get_cifar10_image_datasets(data_dir: Path, train_limit: int | None, test_limit: int | None, seed: int):
    train_set, test_set, _ = get_image_datasets("cifar10", data_dir, train_limit, test_limit, seed)
    return train_set, test_set


def get_cifar10_tensor_datasets(data_dir: Path, train_limit: int | None, test_limit: int | None, seed: int):
    train_set, test_set, _ = get_tensor_datasets("cifar10", data_dir, train_limit, test_limit, seed, "resnet18")
    return train_set, test_set


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def dataset_to_numpy(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for image, label in dataset:
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            arr = np.transpose(arr, (1, 2, 0))
        else:
            arr = np.asarray(image)
        images.append(arr.astype(np.float32))
        labels.append(int(label))
    return np.stack(images), np.asarray(labels, dtype=np.int64)
