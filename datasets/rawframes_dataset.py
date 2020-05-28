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

class RawFramesRecord(object):
    def __init__(self, row):
        self.path = row[0]
        self.num_frames = int(row[1])
        
# If you want testing augmentation, just do it
class RawFramesDataset(Dataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 tmpl='img_{:05d}.jpg',
                 storage_backend='disk',
                 resize=256,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True):

        ann_list = open(ann_file).readlines()
        ann_list = [x.strip().split() for x in ann_list]
        
        self.data = [RawFramesRecord(x) for x in ann_list]
        self.img_prefix = img_prefix
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.tmpl = tmpl
        self.to_rgb = to_rgb
    
        self.storage_backend = storage_backend
        self.mclient, self.cclient = None, None
        self.resize = resize
        loading_funcs = {'disk': self._load_image_disk,
                         'memcached': self._load_image_memcached, 'ceph': self._load_image_ceph}
        self.load_image = loading_funcs[self.storage_backend]

    def __len__(self):
        return len(self.data)

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
        
    def get_frames(self, record):
        pth = record.path
        frames = []
        for i in range(record.num_frames):
            im_name = osp.join(self.img_prefix, pth, self.tmpl.format(i + 1))
            # print(im_name, flush=True)
            frames.append(self.load_image(im_name))
        return frames
        

    def __getitem__(self, idx):
        print('loading image, index is {}'.format(idx), flush=True)
        ims = self.get_frames(self.data[idx])
        try:
            ims = [imresize(im, self.resize) for im in ims]
        except:
            print(idx, self.data[idx].path, self.data[idx].num_frames, flush=True)
            return None
            
        ims = [normalize(im, self.mean, self.std, self.to_rgb) for im in ims]

        ims = [im.transpose(2, 0, 1).astype(np.float32) for im in ims]
        ims = np.stack(ims)
        ims = torch.from_numpy(ims)

        ret = {}
        ret['img'] = ims
        ret['path'] = self.data[idx].path
        return ret
