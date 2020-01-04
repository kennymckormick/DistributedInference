from __future__ import division

import math
import torch
import numpy as np

from torch.distributed import get_world_size, get_rank
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_samples = math.ceil(len(dataset) / samples_per_gpu / num_replicas) * samples_per_gpu
        self.tot_samples = self.num_samples * num_replicas


    def __iter__(self):
        indices = torch.arange(len(self.dataset)).tolist()
        # add more samples to make it divisible
        indices += indices[:(self.tot_samples - len(indices))]
        assert len(indices) == self.tot_samples

        # subsample
        indices = indices[self.rank:self.tot_samples:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
