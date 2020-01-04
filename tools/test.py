import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import time
import torch
import torch.distributed as dist
import os
import os.path as osp
import sys
sys.path = [os.getcwd()] + sys.path
from apis.env import *
import tempfile
import warnings
from utils.dist_utils import get_dist_info
import shutil
import sys
warnings.filterwarnings("ignore", category=UserWarning)
args = None

def multi_test_writebak(model, data_loader, tmpdir='./tmp'):
    model.eval()
    results = []
    rank, world_size = get_dist_info()
    count = 0
    data_time_pool = 0
    proc_time_pool = 0
    tic = time.time()
    for i, data in enumerate(data_loader):
        if i % 100 == 0:
            print('rank {}, data_batch {}'.format(rank, i))
        count = count + 1
        tac = time.time()
        data_time_pool = data_time_pool + tac - tic

        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        toc = time.time()
        proc_time_pool = proc_time_pool + toc - tac

        tic = toc

    print('rank {}, begin collect results'.format(rank), flush=True)
    results = collect_results(results, len(data_loader.dataset), tmpdir)
    return results


def multi_test(model, data_loader, tmpdir='./tmp'):
    model.eval()
    results = []
    rank, world_size = get_dist_info()
    count = 0
    data_time_pool = 0
    proc_time_pool = 0
    tic = time.time()
    for i, data in enumerate(data_loader):
        if i % 100 == 0:
            print('rank {}, data_batch {}'.format(rank, i))
        count = count + 1
        tac = time.time()
        data_time_pool = data_time_pool + tac - tic

        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        toc = time.time()
        proc_time_pool = proc_time_pool + toc - tac

        tic = toc

    print('rank {}, begin collect results'.format(rank), flush=True)
    results = collect_results(results, len(data_loader.dataset), tmpdir)
    return results


def collect_results(result_part, size, tmpdir=None):
    global args

    rank, world_size = get_dist_info()
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        tmpdir = osp.join(tmpdir, args.out.split('.')[0])
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir

    print('rank {} begin dump'.format(rank), flush=True)
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    print('rank {} finished dump'.format(rank), flush=True)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        # shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--testfile', help='checkpoint file', type=str, default='')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--out', help='output result file', type=str, default='default.pkl')
    parser.add_argument('--use_softmax', action='store_true', help='whether to use softmax score')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--fcn_testing', action='store_true')
    parser.add_argument('--flip_testing', action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    global args
    args = parse_args()
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # define your dataset
    # dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    # launcher should be defined
    distributed = True
    init_dist(args.launcher, **cfg.dist_params)

    # define your model
    # model = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)


    # define them on demand
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=1)

    # load weight, may need change
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = DistributedDataParallel(model.cuda())
    outputs = multi_test(model, data_loader)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        # write results

if __name__ == '__main__':
    main()
