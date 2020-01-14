import cv2
import numpy as np
import six


def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)

    
def imfrombytes(content, flag='color'):
    img_np = np.frombuffer(content, np.uint8)
    flag = imread_flags[flag] if is_str(flag) else flag
    img = cv2.imdecode(img_np, flag)
    return img

def imresize(img, size):
    h, w = img.shape[:2]
    if isinstance(size, (float, int)):
        ratio = size / min(h, w)
        newh, neww = int(h * ratio), int(w * ratio)
        size = (neww, newh)
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def imflip(img):
    return np.flip(img, axis=1)

def imcrop(img, size=224, option='M'):
    h, w = img.shape[:2]
    assert h >= size and w >= size
    if option == 'L':
        option = 'LU'
    if option == 'R':
        option = 'RD'
    h_beg, w_beg = (h - size) // 2, (w - size) // 2
    if 'U' in option:
        h_beg = 0
    if 'D' in option:
        h_beg = h - size
    if 'L' in option:
        w_beg = 0
    if 'R' in option:
        w_beg = w - size
    return img[h_beg: h_beg + size, w_beg: w_beg + size]

# this pad is mainly for flow estimation
def impad_to(img, base, padding_val):
    h, w = img.shape[:2]
    newh = int(np.ceil(h / base) * base)
    neww = int(np.ceil(w / base) * base)
    new_img = np.zeros([newh, neww, 3]).astype(np.uint8)
    new_img[:,:] = np.array(padding_val)
    new_img[:h, :w] = img
    return new_img

def normalize(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True):
    if to_rgb:
        img = img[:,:,::-1]
    mean, std = np.array(mean), np.array(std)
    return (img - mean) / std
