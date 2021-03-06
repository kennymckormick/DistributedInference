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
from datasets import build_dataloader, ImageDataset, RawFramesDataset, VideoDataset
from models.resnet import ResNet
from utils.io_utils import load_pickle, dump_pickle
from torch.nn.parallel import DistributedDataParallel
from scipy.special import softmax
warnings.filterwarnings("ignore", category=UserWarning)
args = None

def multi_test(model, data_loader, tmpdir='./tmp'):
    global args
    model.eval()
    results = []
    rank, world_size = get_dist_info()
    print('got dist info', flush=True)
    n_gpu = torch.cuda.device_count()
    my_gpu = rank % n_gpu
    count = 0
    data_time_pool = 0
    proc_time_pool = 0
    tic = time.time()
    for i, data in enumerate(data_loader):
        if i % 100 == 0:
            print('rank {}, data_batch {}'.format(rank, i), flush=True)
            
        count = count + 1
        tac = time.time()
        data_time_pool = data_time_pool + tac - tic
        img, path = data['img'], data['path']
        img = img[0]
        path = path[0]
        num_frames = img.shape[0]
        
        ptr = 0
        results = []
        with torch.no_grad():
            while ptr < num_frames:
                inp = img[ptr: ptr + args.batch_size].to(my_gpu)
                result = model(inp)
                results.append(result.data.cpu().numpy())
                ptr += args.batch_size
        results = np.concatenate(results, axis=0)
        dest_file = osp.join(args.dest, path + '.pkl')
        
        dir_name = osp.dirname(dest_file)
        if not osp.exists(dir_name):
            os.system('mkdir -p ' + dir_name)
            
        dump_pickle(results, dest_file)
        
        toc = time.time()
        proc_time_pool = proc_time_pool + toc - tac
        tic = toc
    
    print('rank {}, finished'.format(rank), flush=True)

# by default, we average on feature generated by fully convolution net to get frame feature
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
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dest', type=str, default='feature')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    global args
    args = parse_args()

    dataset = VideoDataset(args.imglist, args.imgroot)

    # launcher should be defined
    distributed = True
    init_dist(args.launcher, port=args.port)

    # define your model
    model = ResNet(depth=50,
                    num_stages=4,
                    strides=(1, 2, 2, 2),
                    dilations=(1, 1, 1, 1))

    # define them on demand
    data_loader = build_dataloader(dataset, workers_per_gpu=1)

    # load weight, may need change
    model.load_state_dict(torch.load(args.checkpoint), strict=False)

    rank, world_size = get_dist_info()
    n_gpu = torch.cuda.device_count()
    model = model.to(rank % n_gpu)
    
    print('into multi_test', flush=True)

    multi_test(model, data_loader)

if __name__ == '__main__':
    main()
