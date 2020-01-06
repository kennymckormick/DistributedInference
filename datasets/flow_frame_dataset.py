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
class FlowFrameDataset(Dataset):
    def __init__(self,
                 img_list,
                 img_prefix,
                 storage_backend='disk',
                 resize=None,
                 padding_base=1,
                 mean=[0, 0, 0],
                 std=[1, 1, 1],
                 to_rgb=True):

        # param added 9/26/2019, 1:55:42 PM
        self.img_list = img_list
        self.img_prefix = img_prefix

        self.mean = np.array(mean)
        self.std = np.array(std)
        self.to_rgb = to_rgb
        self.padding_base = padding_base

        imgs = open(img_list).read().split('\n')
        imgs = list(map(lambda x: x.split(), imgs))
        self.img_A = list(map(lambda x: osp.join(self.img_prefix, x[0]), imgs))
        self.img_B = list(map(lambda x: osp.join(self.img_prefix, x[1]), imgs))
        self.dest_pth = list(map(lambda x: osp.join(self.img_prefix, x[2]), imgs))
        assert len(self.img_A) == len(self.img_B)


        self.storage_backend = storage_backend
        self.mclient, self.cclient = None, None

        self.resize = resize
        loading_funcs = {'disk': self._load_image_disk,
                         'memcached': self._load_image_memcached, 'ceph': self._load_image_ceph}
        self.load_image = loading_funcs[self.storage_backend]

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
            im = self.load_image(pth)
            if self.resize is not None:
                im = imresize(im, self.resize)

            h, w, _ = im.shape
            # pad pure black
            im = impad_to(im, self.padding_base, [0, 0, 0])
            im = normalize(im, self.mean, self.std, self.to_rgb)
            im = im.transpose(2, 0, 1).astype(np.float32)
            im = torch.from_numpy(im)
            return im, (h, w)

        im_A, org_shape = loadim(self.img_A[idx])
        im_B, org_shape = loadim(self.img_B[idx])
        h, w = org_shape

        im = torch.cat([im_A, im_B], dim=0)
        ret = {}
        ret['img'] = im
        ret['ind'] = idx
        ret['dest'] = self.dest_pth[idx]
        ret['hw'] = np.array([h, w])

        return ret
