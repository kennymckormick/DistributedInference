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
from datasets import build_dataloader, ImageDataset
from models.resnet import ResNet
from utils.io_utils import load_pickle, dump_pickle
from torch.nn.parallel import DistributedDataParallel
from scipy.special import softmax
warnings.filterwarnings("ignore", category=UserWarning)
args = None

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
            result = model(data['img'])
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
        if not osp.exists(tmpdir):
            os.system('mkdir -p ' + tmpdir)
    # dump the part result to the dir

    print('rank {} begin dump'.format(rank), flush=True)
    # also convert to np array
    def tolist(results):
        ret = []
        for item in results:
            item_len = item.shape[0]
            for i in range(item_len):
                ret.append(item[i: i + 1].data.cpu().numpy())
        return ret
    result_part = tolist(result_part)
    dump_pickle(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    print('rank {} finished dump'.format(rank), flush=True)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(load_pickle(part_file))
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
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='weights/resnet50-19c8e357.pth')
    parser.add_argument('--imglist', help='inference list', type=str, default='')
    parser.add_argument('--imgroot', help='data root', type=str, default='')
    parser.add_argument('--port', help='communication port', type=int, default=16807)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--out', help='output result file', type=str, default='default.pkl')
    parser.add_argument('--raw_score', action='store_true', help='whether to use raw score')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--keep_raw', action='store_true', help='keep raw prediction')
    parser.add_argument('--flip_aug', action='store_true')
    parser.add_argument('--crop_aug', type=str, default='None')
    parser.add_argument('--out_pred', help='output predicted label', type=str, default='default.txt')
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
    flip_aug = [False, True] if args.flip_aug else [False]
    crop_aug = ['M']
    if args.crop_aug == 'three':
        crop_aug = ['L', 'M', 'R']
    if args.crop_aug == 'five':
        crop_aug = ['LU', 'RU', 'M', 'LD', 'RD']
    dataset = ImageDataset(args.imglist, args.imgroot, flip_options=flip_aug, crop_options=crop_aug)

    # launcher should be defined
    distributed = True
    init_dist(args.launcher, port=args.port)

    # define your model
    model = ResNet(depth=50,
                    num_stages=4,
                    strides=(1, 2, 2, 2),
                    dilations=(1, 1, 1, 1),
                    style='pytorch',
                    frozen_stages=-1,
                    num_classes=1000)

    # define them on demand
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=2,
        workers_per_gpu=1)

    # load weight, may need change
    model.load_state_dict(torch.load(args.checkpoint))

    model = DistributedDataParallel(model.cuda())
    outputs = multi_test(model, data_loader)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        if not args.raw_score:
            outputs = list(map(softmax, outputs))
        if not args.keep_raw:
            n_aug = len(flip_aug) * len(crop_aug)
            n_samples = len(outputs) // n_aug
            reduced_outputs = list(map(lambda idx:
                        sum(outputs[idx * n_aug: idx * n_aug + n_aug]) / n_aug, range(n_samples)))
            outputs = reduced_outputs
        pred = list(map(np.argmax, outputs))
        if args.out_pred:
            pred_str = [str(x) for x in pred]
            with open(args.out_pred, 'w') as fout:
                fout.write('\n'.join(pred_str))
        dump_pickle(outputs, args.out)

if __name__ == '__main__':
    main()
