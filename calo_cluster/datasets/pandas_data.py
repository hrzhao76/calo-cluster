import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from calo_cluster.datasets.base import BaseDataModule, BaseDataset
from tqdm import tqdm


@dataclass
class PandasDataset(BaseDataset):
    "A generic pandas torch dataset."
    feats: List[str]
    coords: List[str]
    weight: str
    semantic_label: str
    instance_label: str

    def _get_df(self, index: int) -> pd.DataFrame:
        df = pd.read_pickle(self.files[index])
        return df

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        breakpoint()
        df = self._get_df(index)

        #features = df[self.feats].to_numpy(dtype=np.half)
        features = df[self.feats].to_numpy(dtype=np.float32)
        #coordinates = df[self.coords].to_numpy(dtype=np.half)
        coordinates = df[self.coords].to_numpy(dtype=np.float32)

        return_dict = {'features': features, 'coordinates': coordinates}

        if self.semantic_label:
            return_dict['semantic_labels'] = df[self.semantic_label].to_numpy()
        if self.instance_label:
            return_dict['instance_labels'] = df[self.instance_label].to_numpy()
        if self.weight is not None:
            #return_dict['weights'] = df[self.weight].to_numpy(dtype=np.half)
            return_dict['weights'] = df[self.weight].to_numpy(dtype=np.float32)

        return return_dict


@dataclass
class PandasDataModule(BaseDataModule):
    """A generic panda data module that handles common transformations.

    If you set self.transformed_data_dir != self.raw_data_dir in a subclass, and self._transformed_data_dir is empty
    or does not exist, then the function returned by self.get_transform_function will be applied to each raw event
    and the result will be saved to self.transformed_data_dir when prepare_data is called. 

    This allows subclasses to define arbitrary transformations that should be performed before serving data,
    e.g., merging, applying selections, reducing noise levels, etc."""

    data_dir: str
    raw_data_dir: str

    feats: List[str]
    coords: List[str]
    weight: str
    semantic_label: str
    instance_label: str

    def __post_init__(self):
        super().__post_init__()

        self.data_dir = Path(self.data_dir)
        self.raw_data_dir = Path(self.raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.transformed_data_dir = self.raw_data_dir

        self._files = None
        self._raw_files = None

    @property
    def files(self) -> list:
        if self._files is None:
            self._files = []
            self._files.extend(
                sorted(self.transformed_data_dir.glob('*.pkl')))
        return self._files

    @property
    def raw_files(self) -> list:
        if self._raw_files is None:
            self._raw_files = []
            self._raw_files.extend(sorted(self.raw_data_dir.glob('*.pkl')))
        return self._raw_files

    def make_transformed_data(self, ncpus=32):
        transform = self.get_transform_function()
        logging.info(f'Making transformed data at {self.transformed_data_dir}')
        with mp.Pool(ncpus) as p:
            with tqdm(total=len(self.raw_files)) as pbar:
                for _ in p.imap_unordered(transform, self.raw_files):
                    pbar.update()

    def raw_data_exists(self) -> bool:
        return len(set(self.raw_data_dir.glob('*'))) != 0

    def transformed_data_exists(self) -> bool:
        return len(set(self.transformed_data_dir.glob('*'))) != 0

    def prepare_data(self) -> None:
        if not self.transformed_data_exists():
            logging.info(
                f'transformed dataset not found at {self.transformed_data_dir}.')
            if not self.raw_data_exists():
                logging.error(f'Raw dataset not found at {self.raw_data_dir}.')
                raise RuntimeError()
            self.make_transformed_data()

    def get_transform_function(self) -> Callable[[Path], None]:
        """In subclasses, should return a function that accepts an event path as its sole argument."""
        raise NotImplementedError()

    def make_dataset_kwargs(self) -> dict:
        kwargs = {
            'feats': self.feats,
            'coords': self.coords,
            'weight': self.weight,
            'semantic_label': self.semantic_label,
            'instance_label': self.instance_label
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs
