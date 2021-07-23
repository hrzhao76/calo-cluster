import logging
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import uproot
from numpy.core.arrayprint import str_format
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from tqdm import tqdm

from ..utils.comm import get_rank

from typing import Tuple, List

@dataclass
class BaseDataset(Dataset):
    """Base torch dataset."""
    voxel_size: float
    files: List[Path]
    task: str
    feats: List[str] = None
    coords: List[str] = None
    weight: str = None

    scale: bool = False
    std: list = None
    mean: list = None

    def __len__(self):
        return len(self.files)

    def _get_pc_feat_labels(self, index):
        event = pd.read_pickle(self.files[index])
        feat_ = event[self.feats].to_numpy()
        if self.task == 'panoptic':
            labels_ = event[[self.semantic_label, self.instance_label]].to_numpy()
        elif self.task == 'semantic':
            labels_ = event[self.semantic_label].to_numpy()
        elif self.task == 'instance':
            labels_ = event[self.instance_label].to_numpy(
            )
        else:
            raise RuntimeError(f'Unknown task = "{self.task}"')
        pc_ = np.round(event[self.coords].to_numpy() / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        if self.scale:
            feat_ = (feat_ - np.array(self.mean)) / np.array(self.std)

        if self.weight is not None:
            weights_ = event[self.weight].to_numpy()
        else:
            weights_ = None
        return pc_, feat_, labels_, weights_

    def __getitem__(self, index):
        pc_, feat_, labels_, weights_ = self._get_pc_feat_labels(index)
        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)
        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        features = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        inverse_map = SparseTensor(inverse_map, pc_)
        return_dict = {'features': features, 'labels': labels,
                       'inverse_map': inverse_map}
        if weights_ is not None:
            weights = weights_[inds]
            weights = SparseTensor(weights, pc)
            return_dict['weights'] = weights
        return return_dict

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)


@dataclass
class BaseDataModule(pl.LightningDataModule):
    """The base pytorch-lightning data module that handles common data loading tasks.


    This module assumes that the data is organized into a set of files, with one event per file.
    When creating a base class, make sure to override make_dataset appropriately.

    Parameters:
    seed -- a seed used by the RNGs
    task -- the type of ML task that will be performed on this dataset (semantic, instance, panoptic)
    num_epochs -- the number of epochs
    batch_size -- the batch size
    sparse -- whether the data should be provided as SparseTensors (for spvcnn), or not. 
    
    num_workers -- the number of CPU processes to use for data workers.

    event_frac -- the fraction of total data to use
    train_frac -- the fraction of train data to use
    test_frac -- the fraction of test data to use

    cluster_ignore_label -- the semantic label that should be ignored when clustering (needs to be supported by clusterer) and in embed criterion (needs to be supported by embed criterion)
    semantic_ignore_label -- the semantic label that should be ignored in semantic segmentation criterion (needs to be supported by semantic criterion)

    batch_dim -- the dimension that contains batch information, if sparse=False. If sparse=True, the batch should be stored in the last dimension of the coordinates.

    num_classes -- the number of semantic classes
    num_features -- the number of features used as input to the ML model
    voxel_size -- the length of a voxel along one coordinate dimension 
    """

    seed: int
    task: str
    num_epochs: int
    batch_size: int
    sparse: bool

    num_workers: int

    event_frac: float
    train_frac: float
    test_frac: float

    cluster_ignore_label: int
    semantic_ignore_label: int
    
    batch_dim: int

    num_classes: int
    num_features: int
    voxel_size: float


    @property
    def files(self) -> List[Path]:
        raise NotImplementedError()

    def __post_init__(self):
        super().__init__()

        self._validate_fracs()

    def _validate_fracs(self):
        fracs = [self.event_frac, self.train_frac, self.test_frac]
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert self.train_frac + self.test_frac <= 1.0

    def train_val_test_split(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """Returns train, val, and test file lists
        
        Assumes that self.files is defined and there is no preset split in the dataset.
        If the dataset already has train/val/test files defined, override this function
        and return them."""
        files = shuffle(self.files, random_state=42)
        num_files = int(self.event_frac * len(files))
        files = files[:num_files]
        num_train_files = int(self.train_frac * num_files)
        num_test_files = int(self.test_frac * num_files)

        train_files = files[:num_train_files]
        val_files = files[num_train_files:-num_test_files]
        test_files = files[-num_test_files:]

        return train_files, val_files, test_files

    def setup(self, stage: str = None) -> None:
        train_files, val_files, test_files = self.train_val_test_split()

        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)

        if stage == 'fit' or stage is None:
            self.train_dataset = self.make_dataset(train_files)
            self.val_dataset = self.make_dataset(val_files)
        if stage == 'test' or stage is None:
            self.test_dataset = self.make_dataset(test_files)

    def dataloader(self, dataset: BaseDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

    def voxel_occupancy(self) -> np.array:
        """Returns the average voxel occupancy for each batch in the train dataloader."""
        if not self.sparse:
            raise RuntimeError('voxel_occupancy called, but dataset is not sparse!')

        self.batch_size = 1
        dataloader = self.train_dataloader()
        voxel_occupancies = np.zeros(len(dataloader.dataset))
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            voxel_occupancies[i] = len(
                batch['inverse_map'].C) / len(batch['features'].C)

        return voxel_occupancies

    def make_dataset(self, files: List[Path]) -> BaseDataset:
        raise NotImplementedError()
