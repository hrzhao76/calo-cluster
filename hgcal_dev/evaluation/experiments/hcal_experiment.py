from re import L
from hgcal_dev.evaluation.experiments.base_experiment import BaseExperiment, BaseEvent
from hgcal_dev.clustering.meanshift import MeanShift

class HCalEvent(BaseEvent):
    def __init__(self, input_path, instance_label, pred_path=None, task='panoptic'):
        super().__init__(input_path, pred_path=pred_path, class_label='hit', instance_label=instance_label, task=task, clusterer=MeanShift(bandwidth=0.01))

class HCalExperiment(BaseExperiment):
    def __init__(self, wandb_version, ckpt_name=None):
        super().__init__(wandb_version, ckpt_name=ckpt_name)
        if self.cfg.dataset.instance_label == 'truth':
            self.instance_label = 'trackId'
        else:
            self.instance_label = 'RHAntiKtCluster'

    def make_event(self, input_path, pred_path, task):
        return HCalEvent(input_path=input_path, pred_path=pred_path, task=task, instance_label=self.instance_label)