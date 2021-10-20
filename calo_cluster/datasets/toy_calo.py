from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from calo_cluster.datasets.mixins.combine_labels import (
    CombineLabelsDataModuleMixin, CombineLabelsDatasetMixin)
from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)
from calo_cluster.datasets.mixins.toy_calo import ToyCaloDataModuleMixin
from calo_cluster.datasets.pandas_data import PandasDataset


@dataclass
class ToyCaloDataset(SparseDatasetMixin, CombineLabelsDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class ToyCaloOffsetDataset(SparseDatasetMixin, CombineLabelsDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class ToyCaloDataModule(SparseDataModuleMixin, CombineLabelsDataModuleMixin, ScaledDataModuleMixin, ToyCaloDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> ToyCaloDataset:
        kwargs = self.make_dataset_kwargs()
        return ToyCaloDataset(files=files, **kwargs)


@dataclass
class ToyCaloOffsetDataModule(SparseDataModuleMixin, CombineLabelsDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, ToyCaloDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> ToyCaloOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return ToyCaloOffsetDataset(files=files, **kwargs)
