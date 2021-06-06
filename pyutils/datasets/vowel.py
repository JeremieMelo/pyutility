"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 01:08:11
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 01:08:11
"""
from __future__ import print_function

import os
from typing import Any, Callable, Optional, Tuple

import numpy as np

import torch

from torch.functional import Tensor
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_url,
)

__all__ = ["VowelRecognition"]


class VowelRecognition(VisionDataset):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data"
    filename = "vowel-context.data"
    folder = "vowel"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        n_features: int = 10,
        n_labels: int = 10,
        train_ratio: float = 0.7,
        download: bool = False,
    ) -> None:
        root = os.path.join(os.path.expanduser(root), self.folder)
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        super(VowelRecognition, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.train_ratio = train_ratio

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.n_features = n_features
        self.n_labels = n_labels
        assert 1 <= n_features <= 10, print(f"Only support maximum 13 features, but got{n_features}")
        self.data: Any = []
        self.targets = []

        self.process_raw_data()
        self.data, self.targets = self.load(train=train)

    def process_raw_data(self) -> None:
        processed_dir = os.path.join(self.root, "processed")
        processed_training_file = os.path.join(processed_dir, "training.pt")
        processed_test_file = os.path.join(processed_dir, "test.pt")
        if os.path.exists(processed_training_file) and os.path.exists(processed_test_file):
            with open(os.path.join(self.root, "processed/training.pt"), "rb") as f:
                data, targets = torch.load(f)
                if data.shape[-1] == self.n_features:
                    print("Data already processed")
                    return
        data, targets = self._load_dataset()
        data_train, targets_train, data_test, targets_test = self._split_dataset(data, targets)
        data_train, data_test = self._preprocess_dataset(data_train, data_test)
        self._save_dataset(data_train, targets_train, data_test, targets_test, processed_dir)

    def _load_dataset(self) -> Tuple[Tensor, Tensor]:
        data = []
        targets = []
        label_remap = [0, 5, 1, 6, 2, 7, 3, 4, 8, 9, 10]  # the ordering guarantees the task is simple
        select_labels = set(label_remap[: self.n_labels])
        with open(os.path.join(self.root, "raw", self.filename), "r") as f:
            for line in f:
                line = line.strip().split()[3:]
                label = int(line[-1])
                if label not in select_labels:
                    continue
                targets.append(label)
                example = [float(i) for i in line[:-1]]
                data.append(example)

            data = torch.Tensor(data)
            targets = torch.LongTensor(targets)
        return data, targets

    def _split_dataset(self, data: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        from sklearn.model_selection import train_test_split

        data_train, data_test, targets_train, targets_test = train_test_split(
            data, targets, train_size=self.train_ratio, random_state=42
        )
        print(f"training: {data_train.shape[0]} examples, test: {data_test.shape[0]} examples")
        return data_train, targets_train, data_test, targets_test

    def _preprocess_dataset(self, data_train: Tensor, data_test: Tensor) -> Tuple[Tensor, Tensor]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler, RobustScaler

        pca = PCA(n_components=self.n_features)
        data_train_reduced = pca.fit_transform(data_train)
        data_test_reduced = pca.transform(data_test)

        rs = RobustScaler(quantile_range=(10, 90)).fit(
            np.concatenate([data_train_reduced, data_test_reduced], 0)
        )
        data_train_reduced = rs.transform(data_train_reduced)
        data_test_reduced = rs.transform(data_test_reduced)
        mms = MinMaxScaler()
        mms.fit(np.concatenate([data_train_reduced, data_test_reduced], 0))
        data_train_reduced = mms.transform(data_train_reduced)
        data_test_reduced = mms.transform(data_test_reduced)

        return torch.from_numpy(data_train_reduced).float(), torch.from_numpy(data_test_reduced).float()

    def _save_dataset(
        self,
        data_train: Tensor,
        targets_train: Tensor,
        data_test: Tensor,
        targets_test: Tensor,
        processed_dir: str,
    ) -> None:
        try:
            os.mkdir(processed_dir)
        except:
            pass
        processed_training_file = os.path.join(processed_dir, "training.pt")
        processed_test_file = os.path.join(processed_dir, "test.pt")
        with open(processed_training_file, "wb") as f:
            torch.save((data_train, targets_train), f)

        with open(processed_test_file, "wb") as f:
            torch.save((data_test, targets_test), f)
        print(f"Processed dataset saved")

    def load(self, train: bool = True):
        filename = "training.pt" if train else "test.pt"
        with open(os.path.join(self.root, "processed", filename), "rb") as f:
            data, targets = torch.load(f)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
        return data, targets

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_url(self.url, root=os.path.join(self.root, "raw"), filename=self.filename)

    def _check_integrity(self) -> bool:
        return os.path.exists(os.path.join(self.root, "raw", self.filename))

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
