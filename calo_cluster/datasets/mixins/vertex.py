from dataclasses import dataclass

import pandas as pd
import uproot
from calo_cluster.datasets.base import BaseDataModule
from calo_cluster.datasets.pandas_data import PandasDataModuleMixin, PandasDataset
from tqdm.auto import tqdm
import numpy as np

def get_match_idx(truth_flat_event, reco_flat_event):
    match_idx = []
    
    truth_trk_table = truth_flat_event[['d0','z0','phi','theta','qp']].to_numpy()
    reco_trk_table = reco_flat_event[['d0','z0','phi','theta','qp']].to_numpy()
    
    reco_vtxID_list = reco_flat_event['vtxID'].to_numpy()
    
    for truth_trk in truth_trk_table:
        reco_idx = np.flatnonzero((reco_trk_table == truth_trk).all(1))
        if reco_idx.size != 0:
            match_idx.append(reco_vtxID_list[reco_idx[0]]) # Here we just save the first 
        else:
            match_idx.append(-1)
            
    return match_idx


@dataclass
class VertexDatasetMixin(PandasDataset):
    instance_target: str

    def __post_init__(self):
        if self.instance_target == 'reco':
            self.semantic_label = 'reco_semantic_label'
            self.instance_label = 'reco_vtxID'
        elif self.instance_target == 'truth':
            self.semantic_label = 'truth_semantic_label'
            self.instance_label = 'truth_vtxID'
        else:
            raise RuntimeError()
        return super().__post_init__()


@dataclass
class VertexDataModuleMixin(PandasDataModuleMixin, BaseDataModule):
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
            for n in tqdm(range(len(truth_tree[0].array()))):
                truth_df_dict = {k: truth_jagged_dict[k][n] for k in truth_jagged_dict.keys()}
                reco_df_dict = {l: reco_jagged_dict[l][n] for l in reco_jagged_dict.keys()}
                
                truth_flat_event = pd.DataFrame(truth_df_dict)
                reco_flat_event = pd.DataFrame(reco_df_dict)
                truth_flat_event['truth_semantic_label'] = [1] * len(truth_flat_event)
                
                match_idx = get_match_idx(truth_flat_event, reco_flat_event)
                truth_flat_event['reco_AMVF_vtxID'] = match_idx
                truth_flat_event['reco_semantic_label'] = [1] * len(truth_flat_event)
                
                idx_not_found = truth_flat_event['reco_AMVF_vtxID'] == -1
                truth_flat_event.loc[idx_not_found,'reco_semantic_label'] = [0]*len(truth_flat_event['reco_semantic_label'].loc[idx_not_found])
                
                truth_flat_event[coords] /= scale
                truth_flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni += n + 1