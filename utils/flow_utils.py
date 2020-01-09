import os,sys
import os.path as osp
import numpy as np
import cv2

# TVL1 only now
def FlowToImg(raw_flow, bound=0):
    floating_bound = False
    if bound == 0:
        floating_bound = True
    if not floating_bound:
        flow = raw_flow
        flow[flow>bound] = bound
        flow[flow<-bound] = -bound
        flow += bound
        flow *= (255 / float(2*bound))
        flow = flow.astype(np.uint8)
        return flow
    else:
        lb = np.min(raw_flow)
        ub = np.max(raw_flow)
        flow -= lb
        flow *= (255 / (ub - lb))
        flow = flow.astype(np.uint8)
        return flow, lb, ub


def dense_flow(frames):
    n_frame = len(frames)
    gray_frames = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in frames]
    dtvl1=cv2.optflow.DualTVL1OpticalFlow_create()

    gray_st = gray_frames[:-1]
    gray_ed = gray_frames[1: ]

    # shape should be [w, h, 2]
    flowDTVL1 = [dtvl1.calc(x, y, None) for x, y in zip(gray_st, gray_ed)]
    return flowDTVL1

def extract_dense_flow(path, dest, bound=20, write_image=False):
    if osp.exists(path):
        frames = []
        vid = cv2.VideoCapture(path)
        flag, f = vid.read()
        while flag:
            frames.append(f)
            flag, f = vid.read()
    else:
        idx = 0
        im_name = path.format(idx)
        while osp.exists(im_name):
            frames.append(cv2.imread(im_name))
            idx += 1
            im_name = path.format(im_name)
    flow = dense_flow(frames)
    flow_x = [FlowToImg(x[:, :, :1], bound) for x in flow]
    flow_y = [FlowToImg(x[:, :, 1:], bound) for x in flow]
    if not osp.exists(dest):
        os.system('mkdir -p ' + dest)
    flow_x_names = [osp.join(dest, 'x_{:05d}.jpg'.format(ind)) for ind in range(len(flow_x))]
    flow_y_names = [osp.join(dest, 'y_{:05d}.jpg'.format(ind)) for ind in range(len(flow_y))]

    for imx, namex in zip(flow_x, flow_x_names):
        cv2.imwrite(namex, imx)
    for imy, namey in zip(flow_y, flow_y_names):
        cv2.imwrite(namey, imy)
    if write_image:
        im_names = [osp.join(dest, 'img_{:05d}.jpg'.format(ind)) for ind in range(len(frames))]
        for im, name in zip(frames, im_names):
            cv2.imwrite(name, im)


def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image
    Args:
        flow(ndarray): optical flow
        color_wheel(ndarray or None): color wheel used to map flow field to RGB
            colorspace. Default color wheel will be used if not specified
        unknown_thr(str): values above this threshold will be marked as unknown
            and thus ignored

    Returns:
        ndarray: an RGB image that can be visualized
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
                   (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    [h, w] = dx.shape

    rad = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(-dy, -dx) / np.pi

    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (
        1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]
    small_ind = rad <= 1
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0
    flow_img = (flow_img * 255).astype(np.uint8)

    return flow_img


def make_color_wheel(bins=None):
    """Build a color wheel
    Args:
        bins(list or tuple, optional): specify number of bins for each color
            range, corresponding to six ranges: red -> yellow, yellow -> green,
            green -> cyan, cyan -> blue, blue -> magenta, magenta -> red.
            [15, 6, 4, 11, 13, 6] is used for default (see Middlebury).

    Returns:
        ndarray: color wheel of shape (total_bins, 3)
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)

    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]

    num_bins = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros((3, num_bins), dtype=np.float32)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T
