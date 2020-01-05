import os,sys
import os.path as osp
import numpy as np
import cv2

# TVL1 only now
def FlowToImg(raw_flow, bound=20):
    flow = raw_flow
    flow[flow>bound] = bound
    flow[flow<-bound] = -bound
    flow += bound
    flow *= (255 / float(2*bound))
    flow = flow.astype(np.uint8)


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
