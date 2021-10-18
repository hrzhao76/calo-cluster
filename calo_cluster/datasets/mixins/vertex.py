from dataclasses import dataclass

import pandas as pd
import uproot
from calo_cluster.datasets.base import BaseDataModule
from calo_cluster.datasets.pandas_data import PandasDataset
from tqdm import tqdm


@dataclass
class VertexDatasetMixin(PandasDataset):
    instance_target: str

    def __post_init__(self):
        if self.instance_target == 'reco':
            self.semantic_label = 'semantic_label'
            self.instance_label = 'reco_vtxID'
        elif self.instance_target == 'truth':
            raise NotImplementedError()
        else:
            raise RuntimeError()
        return super().__post_init__()


def get_match_idx(truth_z0_list, reco_z0_list, reco_vtxID_list):
    match_idx = []
    for truth_z0 in truth_z0_list:
        reco_idx = np.where(reco_z0_list == truth_z0)
        if reco_idx[0].size != 0:
            match_idx.append(reco_vtxID_list[reco_idx[0][0]]) # Here we just save the first 
        else:
            match_idx.append(-1)
            
    return match_idx


@dataclass
class VertexDataModuleMixin(BaseDataModule):
    instance_target: str

    def make_dataset_kwargs(self) -> dict:
        kwargs = {
            'instance_target': self.instance_target
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs

    @staticmethod
    def root_to_pickle(root_data_path, raw_data_dir):
        ni = 0
        for f in sorted(root_data_path.glob('*.root')):
            root_dir = uproot.open(f)
            truth_tree = root_dir['Truth_Vertex_PV_Selected']
            reco_tree = root_dir['Reco_Vertex']
            truth_jagged_dict = {}
            reco_jagged_dict = {}
            truth_prefix = 'truth_vtx_fitted_trk_'
            reco_prefix = 'reco_vtx_fitted_trk_'

            for k, v in tqdm(truth_tree.items()):
                if not k.startswith(truth_prefix):
                    continue
                truth_jagged_dict[k[len(truth_prefix):]] = v.array()
            
            for k, v in tqdm(reco_tree.items()):
                if not k.startswith(reco_prefix):
                    continue
                reco_jagged_dict[k[len(reco_prefix):]] = v.array()
            
            truth_jagged_dict['truth_vtxID'] = truth_jagged_dict.pop('vtxID')

            coords = ['d0', 'z0', 'phi', 'theta', 'qp']
            scale = np.array([0.05, 500, 6, 2, 4])
            for n in tqdm(range(len(reco_tree[0].array()))):
                df_dict = {k: truth_jagged_dict[k][n] for k in truth_jagged_dict.keys()}
                flat_event = pd.DataFrame(df_dict)
                match_idx = get_match_idx(np.array(truth_jagged_dict['z0'][n]), np.array(reco_jagged_dict['z0'][n]), np.array(reco_jagged_dict['vtxID'][n]))
                flat_event['AMVF_reco_ID'] = match_idx
                flat_event[coords] /= scale
                flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni += n + 1
