import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import torch
import os
from .utils import imfrombytes, imresize, imflip, imcrop, normalize, impad_to
import sys
import cv2
import random as rd
sys.path.append('/mnt/lustre/share/pymc/py3')
try:
    import mc
    import ceph
except ImportError:
    pass


# By default, it will reture 6 channel image
# one line of video_list looks like: video_pth flow_tmpl({x/y}_{index})
class FlowVideoDataset(Dataset):
    def __init__(self,
                 video_list,
                 video_prefix,
                 resize=None,
                 padding_base=1,
                 mean=0,
                 std=1,
                 to_rgb=True):

        # param added 9/26/2019, 1:55:42 PM
        self.video_list = video_list
        self.video_prefix = video_prefix

        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
        self.padding_base = padding_base

        videos = open(video_list).read().split('\n')
        videos = list(map(lambda x: x.split(), videos))

        self.videos = list(map(lambda x: osp.join(self.img_prefix, x[0]), videos))
        self.tmpls = list(map(lambda x: osp.join(self.img_prefix, x[1]), videos))
        assert len(self.videos) == len(self.tmpls)

        self.resize = resize

    def __len__(self):
        return len(self.videos)

    # use opencv
    def load_video(self, pth):
        vid = cv2.VideoCapture(pth)
        frames = []
        flag, f = vid.read()
        while flag:
            frames.append(f)
            flag, f = vid.read()
        return frames


    def __getitem__(self, idx):
        def loadvid(pth):
            ims = self.loadvid(pth)
            if self.resize is not None:
                ims = list(map(lambda im: imresize(im, self.resize), ims))

            h, w, _ = ims[0].shape
            # pad pure black
            if self.padding_base > 1:
                ims = [impad_to(im, self.padding_base, [0, 0, 0]) for im in ims]
            if self.mean != 0 or self.std != 1:
                ims = [normalize(im, self.mean, self.std, self.to_rgb) for im in ims]

            ims = [im.transpose(2, 0, 1).astype(np.float32) for im in ims]
            ims = np.stack(ims, axis=0)
            ims = torch.from_numpy(ims)
            return ims, (h, w)

        ims, org_shape = loadvid(self.videos[idx])
        h, w = org_shape

        ret = {}
        ret['im_A'] = ims[:-1]
        ret['im_B'] = ims[1: ]
        ret['ind'] = idx
        ret['dest'] = self.tmpls[idx]
        ret['hw'] = np.array([h, w])

        return ret
