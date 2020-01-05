import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import torch
import os
from .utils import imfrombytes, imresize, imflip, imcrop, normalize
import sys
import cv2
import random as rd
sys.path.append('/mnt/lustre/share/pymc/py3')
try:
    import mc
    import ceph
except ImportError:
    pass

# If you want testing augmentation, just do it
class ImageDataset(Dataset):
    def __init__(self,
                 img_list,
                 img_prefix,
                 storage_backend='disk',
                 # rescale can be int(short edge) or tuple(w, h)
                 resize=256,
                 flip_options=[False],
                 crop_size=224,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True,
                 # valid crop_options: LU, RU, L, R, M, LD, RD
                 crop_options=['M']):

        # param added 9/26/2019, 1:55:42 PM
        self.img_list = img_list
        self.img_prefix = img_prefix

        self.mean = np.array(mean)
        self.std = np.array(std)
        self.to_rgb = to_rgb

        imgs = open(img_list).read().split('\n')
        while '' == imgs[-1]:
            imgs = imgs[:-1]
        self.imgs = list(map(lambda x: osp.join(img_prefix, x), imgs))
        self.storage_backend = storage_backend
        self.mclient, self.cclient = None, None

        self.flip_options = flip_options
        self.crop_options = crop_options
        self.n_aug = len(flip_options) * len(crop_options)
        self.resize = resize
        self.crop_size = crop_size
        loading_funcs = {'disk': self._load_image_disk,
                         'memcached': self._load_image_memcached, 'ceph': self._load_image_ceph}
        self.load_image = loading_funcs[self.storage_backend]

    def __len__(self):
        return len(self.imgs) * self.n_aug

    def _ensure_memcached(self):
        if self.mclient is None:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
        return

    def _ensure_ceph(self):
        if self.cclient is None:
            self.cclient = ceph.S3Client()
        return

    def _load_image_disk(self, pth):
        return cv2.imread(pth)

    def _load_image_memcached(self, pth):
        self._ensure_memcached()
        value = mc.pyvector()
        self.mclient.Get(pth, value)
        value_buf = mc.ConvertBuffer(value)
        return imfrombytes(value_buf)

    def _load_image_ceph(self, pth):
        self._ensure_ceph()
        value = self.cclient.Get(pth)
        value_buf = memoryview(value)
        return imfrombytes(value_buf)

    def __getitem__(self, idx):
        im_idx = idx // self.n_aug
        aug_idx = idx % self.n_aug
        flip_idx = aug_idx // len(self.crop_options)
        crop_idx = aug_idx % len(self.crop_options)

        flip_opt = self.flip_options[flip_idx]
        crop_opt = self.crop_options[crop_idx]
        im = self.load_image(self.imgs[im_idx])
        try:
            im = imresize(im, self.resize)
        except:
            print(idx)
        if flip_opt:
            im = imflip(im)
        im = imcrop(im, self.crop_size, crop_opt)
        im = normalize(im, self.mean, self.std, self.to_rgb)

        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im)

        ret = {}
        ret['img'] = im
        ret['ind'] = idx
        return ret
