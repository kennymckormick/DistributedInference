import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import torch
import os
from .utils import imfrombytes, imresize, imflip, imcrop, normalize
import sys
import cv2
import random as rd
import decord
sys.path.append('/mnt/lustre/share/pymc/py3')

# If you want testing augmentation, just do it
class VideoDataset(Dataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 resize=256,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):

        # just video names
        ann_list = open(ann_file).readlines()
        ann_list = [x.strip() for x in ann_list]
        
        self.data = ann_list
        self.img_prefix = img_prefix
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.resize = resize

    def __len__(self):
        return len(self.data)
        
    def get_frames(self, video):
        pth = osp.join(self.img_prefix, video)
        vid = decord.VideoReader(pth)
        num_frame = len(vid)
        frames = [vid[i].asnumpy() for i in range(num_frame)]
        return frames
        

    def __getitem__(self, idx):
        ims = self.get_frames(self.data[idx])
        try:
            ims = [imresize(im, self.resize) for im in ims]
        except:
            print(idx, self.data[idx], flush=True)
            return None
            
        ims = [normalize(im, self.mean, self.std, False) for im in ims]
        ims = [im.transpose(2, 0, 1).astype(np.float32) for im in ims]
        ims = np.stack(ims)
        ims = torch.from_numpy(ims)

        ret = {}
        ret['img'] = ims
        ret['path'] = self.data[idx]
        return ret