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
class MiniVideoDataset(Dataset):
    def __init__(self,
                 path,
                 rgb_root,
                 flow_root,
                 tmpl=None,
                 storage_backend='disk',
                 resize=None,
                 padding_base=1,
                 mean=0,
                 std=1,
                 to_rgb=True):

        # param added 9/26/2019, 1:55:42 PM
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        self.path = path
        self.tmpl = tmpl
        self.storage_backend = storage_backend

        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
        self.padding_base = padding_base

        self.num_frames = len(os.listdir(osp.join(self.rgb_root, path)))
        self.frames = list(map(lambda x: osp.join(self.rgb_root, self.path,
                                    tmpl.format(x + 1)), range(self.num_frames)))

        self.img_A = self.frames[:-1]
        self.img_B = self.frames[1:]

        self.dest_pth = list(map(lambda x: osp.join(self.flow_root, self.path,
                                    '{}_{:05d}.jpg'.format('{}', x + 1)),
                                    range(self.num_frames - 1)))

        assert len(self.img_A) == len(self.img_B)


        self.storage_backend = storage_backend
        self.mclient, self.cclient = None, None

        self.resize = resize
        loading_funcs = {'disk': self._load_image_disk,
                         'memcached': self._load_image_memcached, 'ceph': self._load_image_ceph}
        self.load_image = loading_funcs[self.storage_backend]
        self.cache = {}

    def __len__(self):
        return len(self.img_A)

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
        def loadim(pth):
            if pth in self.cache:
                ret = self.cache.pop(pth)
                return ret
            im = self.load_image(pth)
            h, w, _ = im.shape

            if self.resize is not None:
                im = imresize(im, self.resize)

            # pad pure black
            im = impad_to(im, self.padding_base, [0, 0, 0])
            if self.mean == 0 and self.std == 1:
                pass
            else:
                im = normalize(im, self.mean, self.std, self.to_rgb)
            im = im.transpose(2, 0, 1).astype(np.float32)
            im = torch.from_numpy(im)
            self.cache[pth] = (im, (h, w))
            return im, (h, w)

        im_A, org_shape = loadim(self.img_A[idx])
        im_B, org_shape = loadim(self.img_B[idx])
        h, w = org_shape

        ret = {}
        ret['im_A'] = im_A
        ret['im_B'] = im_B
        ret['ind'] = idx
        ret['dest'] = self.dest_pth[idx]
        ret['hw'] = np.array([h, w])
        return ret
