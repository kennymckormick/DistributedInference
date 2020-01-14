import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import time
import torch
import torch.distributed as dist
import os
import os.path as osp
import sys
import cv2
sys.path = [os.getcwd()] + sys.path
from apis.env import *
import tempfile
import warnings
from utils.dist_utils import get_dist_info
import shutil
import sys
from datasets import FlowFrameDataset, FlowVideoDataset, MiniVideoDataset
from utils.io_utils import load_pickle, dump_pickle
from torch.nn.parallel import DistributedDataParallel
from scipy.special import softmax
from utils.flow_utils import FlowToImg, flow2rgb, prenorm
from utils.io_utils import mrlines
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate
from abc import abstractproperty as ABC

# for pwcnet, just store the 160p image ...
# input: 320p, resize: 640p, output: 160p

warnings.filterwarnings("ignore", category=UserWarning)
args = None

# from models.flownet2 import FlowNet2
from models.pwcnet import PWCNet
from models.vcn import VCN

class MyDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.len = len(dataset)
        self.batch_size = batch_size
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr == self.len:
            raise StopIteration
        ed = self.ptr + batch_size
        ed = self.len if ed > self.len else ed
        data = []
        for i in range(self.ptr, ed):
            data.append(self.dataset.__getitem__(i))
        self.ptr = ed
        return collate.default_collate(data)



def multi_test_minivideo(model, jobs, bound=0):
    algo = args.algo
    vis = args.vis
    out_flo = args.out_flo
    to_rgb = args.to_rgb
    mean = args.mean
    std = args.std
    batch_size = args.batch_size
    src = args.src
    dest = args.dest
    se = args.se
    out_se = args.out_se
    padding_base=args.pad_base

    model.eval()
    results = []
    rank, world_size = get_dist_info()
    n_gpu = torch.cuda.device_count()
    my_gpu = rank % n_gpu

    # batch size of outer loop is always 1

    for i, job in enumerate(jobs):
        logfile = osp.join(dest.replace('flow', 'flow_info'), job + '.txt')
        base = osp.dirname(logfile)
        if not osp.exists(base):
            os.system('mkdir -p ' + base)
        if i % 10 == 0:
            print('rank {}, data_batch {}'.format(rank, i))
        mini_dataset = MiniVideoDataset(job, src, dest,
                tmpl='img_{:05d}.jpg', storage_backend='memcached',
                resize=se, padding_base=padding_base,
                mean=mean, std=std, to_rgb=to_rgb)
        num_frames = mini_dataset.num_frames
        mini_loader = MyDataLoader(mini_dataset, batch_size)

        fout = open(logfile, 'w')
        with torch.no_grad():
            for j, data in enumerate(mini_loader):
                im_A, im_B = data['im_A'], data['im_B']
                im_A, im_B = im_A.to(my_gpu), im_B.to(my_gpu)

                dests = data['dest']
                hws = data['hw'].data.cpu().numpy()

                result = model(im_A, im_B)
                if algo == 'pwcnet':
                    result = result * 5.0
                    # output 4x resolution lower
                if algo == 'vcn':
                    result = result[0]

                result = result.data.cpu().numpy()
                this_batch = im_A.shape[0]
                for k in range(this_batch):
                    h, w = hws[k][0], hws[k][1]
                    this_dest = dests[k]
                    flow = result[k].transpose(1,2,0)

                    if args.algo == 'pwcnet':
                        pre_se = min(flow.shape[:2])
                        preh, prew, _ = flow.shape
                        if out_se != 0:
                            post_factor = out_se / pre_se
                            posth, postw = int(post_factor * preh), int(post_factor * prew)
                        else:
                            post_factor = min(h, w) / min(preh, prew)
                            posth, postw = h, w
                        if post_factor != 1:
                            flow = cv2.resize(flow, (postw, posth))
                            flow *= post_factor
                    else:
                        raise NotImplementedError
                    flow, norm = prenorm(flow, 32.0)
                    flow_x, lb_x, ub_x = FlowToImg(flow[:,:,:1], bound)
                    flow_y, lb_y, ub_y = FlowToImg(flow[:,:,1:], bound)

                    base_pth = osp.dirname(this_dest)
                    if not osp.exists(base_pth):
                        os.system('mkdir -p ' + base_pth)
                    flow_x_name = this_dest.format('x')
                    flow_y_name = this_dest.format('y')
                    cv2.imwrite(flow_x_name, flow_x)
                    cv2.imwrite(flow_y_name, flow_y)
                    fout.write('{} {:.4f} {:.4f}\n'.format(flow_x_name, lb_x, ub_x))
                    fout.write('{} {:.4f} {:.4f}\n'.format(flow_y_name, lb_y, ub_y))
        fout.close()


# By default, use dyna flow, for each video, also output a list
# for flow frame testing, output to `log/{rank}.txt`
def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    # should be youtube_ids
    parser.add_argument('--list', help='inference list', type=str)
    parser.add_argument('--src', help='frames root dir', type=str)
    parser.add_argument('--dest', help='flow dest dir', type=str)
    parser.add_argument('--port', help='communication port', type=int, default=16807)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--pad_base', type=int, default=64)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--algo', type=str, help='algorithm to use for flow estimation', default='pwcnet')
    parser.add_argument('--out_flo', action='store_true')
    # set edge length of short edge
    parser.add_argument('--se', type=int, default=640)
    parser.add_argument('--out_se', type=int, default=160)
    parser.add_argument('--batch_size', help='batch_size in video', type=int, default=4)
    parser.add_argument('--store', help='store rgb frames extracted', action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    global args
    args = parse_args()

    to_rgb = True
    std = 1.0
    mean = 0.0
    if args.algo == 'pwcnet' or args.algo == 'vcn':
        to_rgb = False
        std = 255.0
    if args.algo == 'vcn':
        mean = [83.33498422,  93.08780475, 101.84256047]

    args.to_rgb = to_rgb
    args.std = std
    args.mean = mean

    assert (not args.vis)
    assert (not args.out_flo)

    distributed = True
    init_dist(args.launcher, port=args.port)


    if args.algo == 'flownet2':
        args.checkpoint = 'weights/FlowNet2_checkpoint.pth.tar'
        model_args = ABC()
        model_args.fp16 = False
        model_args.rgb_max = 255.0
        model = FlowNet2(model_args)
    elif args.algo == 'pwcnet':
        args.checkpoint = 'weights/pwcnet.pth.tar'
        model = PWCNet()
    elif args.algo == 'vcn':
        args.checkpoint = 'weights/vcn.pth.tar'
        model = VCN()
    else:
        raise NotImplementedError('algorithm not supported')

    state_dict = torch.load(args.checkpoint)

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # load weight, may need change
    model.load_state_dict(state_dict)
    rank, world_size = get_dist_info()

    n_gpu = torch.cuda.device_count()

    model = model.to(rank % n_gpu)

    all_work = mrlines(args.list)
    my_job = all_work[rank::world_size]
    outputs = multi_test_minivideo(model, my_job)

if __name__ == '__main__':
    main()
