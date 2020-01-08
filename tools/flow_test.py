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
from datasets import build_dataloader, FlowFrameDataset
from utils.io_utils import load_pickle, dump_pickle
from torch.nn.parallel import DistributedDataParallel
from scipy.special import softmax
from utils.flow_utils import FlowToImg, flow2rgb
from abc import abstractproperty as ABC

warnings.filterwarnings("ignore", category=UserWarning)
args = None

from models.flownet2 import FlowNet2
from models.pwcnet import PWCNet
from models.vcn import VCN

# pwcnet downsample by 4 ...
def multi_test_writebak(model, data_loader, tmpdir='./tmp', bound=20.0):
    algo = args.algo
    vis = args.vis
    out_flo = args.out_flo
    model.eval()
    results = []
    rank, world_size = get_dist_info()
    n_gpu = torch.cuda.device_count()
    my_gpu = rank % n_gpu
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
            inp1, inp2 = data['im_A'].to(my_gpu), data['im_B'].to(my_gpu)
            result = model(inp1, inp2)

            if algo == 'pwcnet':
                result = result * 20.0
            if algo == 'vcn':
                result = result[0]

            names = data['dest']
            result = result.data.cpu().numpy()

            batch_size = len(names)
            hws = data['hw'].data.cpu().numpy()
            # for image with different shape, batch_size = 1
            # if you want a larger batch_size, your input image shape should be exactly same
            # you can improve it by using group sampler, but im lazy ...
            for i in range(batch_size):
                tmpl = names[i]
                hw = hws[i]
                h, w = hw[0], hw[1]
                flow = result[i].transpose(1, 2, 0)
                if args.se != 0:
                    prescale_factor = args.se / min(h, w)
                    preh, prew = int(prescale_factor * h), int(prescale_factor * w)
                else:
                    preh, prew = h, w
                flow = flow[:preh, :prew]
                if args.out_se != 0:
                    postscale_factor = args.out_se / min(preh, prew)
                    posth, postw = int(postscale_factor * preh), int(postscale_factor * prew)
                else:
                    postscale_factor = min(h, w) / min(preh, prew)
                    posth, postw = h, w
                flow = cv2.resize(flow, (postw, posth))
                flow *= postscale_factor

                if not vis:
                    if out_flo:
                        base_pth = osp.dirname(tmpl)
                        if not osp.exists(base_pth):
                            os.system('mkdir -p ' + base_pth)
                        np.save(tmpl.format('flo').replace('jpg', 'npy'), flow)
                    else:
                        flow_x = FlowToImg(flow[:,:,:1])
                        flow_y = FlowToImg(flow[:,:,1:])
                        base_pth = osp.dirname(tmpl)
                        if not osp.exists(base_pth):
                            os.system('mkdir -p ' + base_pth)
                        cv2.imwrite(tmpl.format('x'), flow_x)
                        cv2.imwrite(tmpl.format('y'), flow_y)
                else:
                    img = flow2rgb(flow)
                    base_pth = osp.dirname(tmpl)
                    if not osp.exists(base_pth):
                        os.system('mkdir -p ' + base_pth)
                    cv2.imwrite(tmpl.format('vis'), img[:,:,::-1])
        toc = time.time()
        proc_time_pool = proc_time_pool + toc - tac
        tic = toc



def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--imglist', help='inference list', type=str, default='')
    parser.add_argument('--imgroot', help='data root', type=str, default='')
    parser.add_argument('--port', help='communication port', type=int, default=16807)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--pad_base', type=int, default=None)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--algo', type=str, help='algorithm to use for flow estimation', default='flownet2')
    parser.add_argument('--out_flo', action='store_true')
    # set edge length of short edge
    parser.add_argument('--se', type=int, default=480)
    parser.add_argument('--out_se', type=int, default=240)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    global args
    args = parse_args()

    to_rgb = True
    std = 1.0
    if args.algo == 'pwcnet' or args.algo == 'vcn':
        to_rgb = False
        std = 255.0
    if args.algo == 'vcn':
        mean = [83.33498422,  93.08780475, 101.84256047]
        
    dataset = FlowFrameDataset(args.imglist, args.imgroot, padding_base=args.pad_base, to_rgb=to_rgb,
                                                    std=std, mean=mean, resize=args.se)
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

    # define them on demand
    # By default, flow frame loader have batch size 1, in case different shape
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=1)

    # load weight, may need change
    model.load_state_dict(state_dict)
    rank, world_size = get_dist_info()
    n_gpu = torch.cuda.device_count()

    model = model.to(rank % n_gpu)
    outputs = multi_test_writebak(model, data_loader)

if __name__ == '__main__':
    main()
