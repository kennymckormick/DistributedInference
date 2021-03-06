from functools import partial

from utils.dist_utils import get_dist_info
from torch.utils.data import DataLoader

# Do not use mmcv collate

from .sampler import DistributedSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
if rlimit[0] < 4096:
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# No shuffle
def build_dataloader(dataset,
                     imgs_per_gpu=1,
                     workers_per_gpu=1,
                     rank_tot = None,
                     **kwargs):

    if rank_tot == None:
        rank, world_size = get_dist_info()
    else:
        rank, world_size = rank_tot
    sampler = DistributedSampler(dataset, imgs_per_gpu, world_size, rank)
    batch_size = imgs_per_gpu
    num_workers = workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        **kwargs)

    return data_loader
