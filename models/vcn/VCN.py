"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory
import numpy as np
import math
import pdb
import time

from .submodule import pspnet, conv
from .conv4d import sepConv4d, butterfly4D



# Since flow_reg and warpmodule do not need any parameter, use function instead of class
# since use function instead of class, new tensor on input device
def flow_reg(x, ent=False, maxdisp=int(4)):
    # Begin definition
    b,u,v,h,w = x.shape
    md = maxdisp
    truncated = True
    wsize = 3
    device = x.device
    flowrangey = range(-maxdisp, maxdisp + 1)
    flowrangex = range(-maxdisp, maxdisp + 1)
    meshgrid = np.meshgrid(flowrangex,flowrangey)
    flowy = np.tile( np.reshape(meshgrid[0],[1,2*maxdisp+1,2*maxdisp+1,1,1]), (b,1,1,h,w) )
    flowx = np.tile( np.reshape(meshgrid[1],[1,2*maxdisp+1,2*maxdisp+1,1,1]), (b,1,1,h,w) )
    flowx, flowy = torch.Tensor(flowx).to(device), torch.Tensor(flowy).to(device)
    pool3d = nn.MaxPool3d((wsize*2+1,wsize*2+1,1),stride=1,padding=(wsize,wsize,0))
    # Begin operation

    oldx = x
    x = x.view(b,u*v,h,w)

    idx = x.argmax(1)[:,np.newaxis]
    if x.is_cuda:
        mask = torch.cuda.HalfTensor(b,u*v,h,w).fill_(0).to(device)
    else:
        mask = torch.FloatTensor(b,u*v,h,w).fill_(0)
    mask.scatter_(1,idx,1)
    mask = mask.view(b,1,u,v,-1)
    mask = pool3d(mask)[:,0].view(b,u,v,h,w)

    ninf = x.clone().fill_(-np.inf).view(b,u,v,h,w)
    x = torch.where(mask.byte(),oldx,ninf)

    b,u,v,h,w = x.shape
    x = F.softmax(x.view(b,-1,h,w),1).view(b,u,v,h,w)
    outx = torch.sum(torch.sum(x*flowx,1),1,keepdim=True)
    outy = torch.sum(torch.sum(x*flowy,1),1,keepdim=True)

    if ent:
        local_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
        if wsize == 0:
            local_entropy[:] = 1.
        else:
            local_entropy /= np.log((wsize*2+1)**2)

        x = F.softmax(oldx.view(b,-1,h,w),1).view(b,u,v,h,w)
        global_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
        global_entropy /= np.log(x.shape[1]*x.shape[2])
        return torch.cat([outx,outy],1),torch.cat([local_entropy, global_entropy],1)
    else:
        return torch.cat([outx,outy],1),None


def warp(x, flo):
    b, c, h, w = x.shape
    device = x.device
    xx = torch.arange(0, w).view(1,-1).repeat(h,1).to(device)
    yy = torch.arange(0, h).view(-1,1).repeat(1,w).to(device)
    xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flo
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(w-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(h-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = ((vgrid[:,:,:,0].abs()<1) * (vgrid[:,:,:,1].abs()<1)) >0
    return output*mask.unsqueeze(1).float(), mask



class VCN(nn.Module):
    """
    VCN.
    md defines maximum displacement for each level, following a coarse-to-fine-warping scheme
    fac defines squeeze parameter for the coarsest level
    """
    def __init__(self, md=[4,4,4,4,4]):
        super(VCN,self).__init__()
        self.md = md
        use_entropy = True
        withbn = True

        ## pspnet
        self.pspnet = pspnet(is_proj=False)

        ## Volumetric-UNet
        fdima1 = 128 # 6/5/4
        fdima2 = 64 # 3/2
        fdimb1 = 16 # 6/5/4/3
        fdimb2 = 12 # 2

        full=False
        self.f6 = butterfly4D(fdima1, fdimb1,withbn=withbn,full=full)
        self.p6 = sepConv4d(fdimb1,fdimb1, with_bn=False, full=full)

        self.f5 = butterfly4D(fdima1, fdimb1,withbn=withbn, full=full)
        self.p5 = sepConv4d(fdimb1,fdimb1, with_bn=False,full=full)

        self.f4 = butterfly4D(fdima1, fdimb1,withbn=withbn,full=full)
        self.p4 = sepConv4d(fdimb1,fdimb1, with_bn=False,full=full)

        self.f3 = butterfly4D(fdima2, fdimb1,withbn=withbn,full=full)
        self.p3 = sepConv4d(fdimb1,fdimb1, with_bn=False,full=full)

        full=True
        self.f2 = butterfly4D(fdima2, fdimb2,withbn=withbn,full=full)
        self.p2 = sepConv4d(fdimb2,fdimb2, with_bn=False,full=full)

        ## soft WTA modules
        self.flow_reg64 = lambda x: flow_reg(x, ent=use_entropy, maxdisp=self.md[0])
        self.flow_reg32 = lambda x: flow_reg(x, ent=use_entropy, maxdisp=self.md[1])
        self.flow_reg16 = lambda x: flow_reg(x, ent=use_entropy, maxdisp=self.md[2])
        self.flow_reg8 =  lambda x: flow_reg(x, ent=use_entropy, maxdisp=self.md[3])
        self.flow_reg4 =  lambda x: flow_reg(x, ent=use_entropy, maxdisp=self.md[4])

        # function to replace module
        self.warp5 = warp
        self.warp4 = warp
        self.warp3 = warp
        self.warp2 = warp


        ## hypotheses fusion modules, adopted from the refinement module of PWCNet
        # https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
        # c6
        self.dc6_conv1 = conv(128+4*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc6_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc6_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc6_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc6_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc6_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc6_conv7 = nn.Conv2d(32,2*fdimb1,kernel_size=3,stride=1,padding=1,bias=True)

        # c5
        self.dc5_conv1 = conv(128+4*fdimb1*2, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc5_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc5_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc5_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc5_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc5_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc5_conv7 = nn.Conv2d(32,2*fdimb1*2,kernel_size=3,stride=1,padding=1,bias=True)

        # c4
        self.dc4_conv1 = conv(128+4*fdimb1*3, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc4_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc4_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc4_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc4_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc4_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc4_conv7 = nn.Conv2d(32,2*fdimb1*3,kernel_size=3,stride=1,padding=1,bias=True)

        # c3
        self.dc3_conv1 = conv(64+16*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc3_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc3_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc3_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc3_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc3_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc3_conv7 = nn.Conv2d(32,8*fdimb1,kernel_size=3,stride=1,padding=1,bias=True)
        # c2
        self.dc2_conv1 = conv(64+16*fdimb1+4*fdimb2, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc2_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc2_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc2_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc2_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc2_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc2_conv7 = nn.Conv2d(32,4*2*fdimb1 + 2*fdimb2,kernel_size=3,stride=1,padding=1,bias=True)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()


    def corrf(self, refimg_fea, targetimg_fea,maxdisp):
        """
        another correlation function giving the same result as corr()
        supports backwards
        """
        b,c,height,width = refimg_fea.shape
        if refimg_fea.is_cuda:
            cost = Variable(torch.cuda.FloatTensor(b,c,2*maxdisp+1,2*maxdisp+1,height,width)).fill_(0.) # b,c,u,v,h,w
        else:
            cost = Variable(torch.FloatTensor(b,c,2*maxdisp+1,2*maxdisp+1,height,width)).fill_(0.) # b,c,u,v,h,w
        for i in range(2*maxdisp+1):
            ind = i-maxdisp
            for j in range(2*maxdisp+1):
                indd = j-maxdisp
                feata = refimg_fea[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                featb = targetimg_fea[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]
                diff = (feata*featb)
                cost[:, :, i,j,max(0,-indd):height-indd,max(0,-ind):width-ind]   = diff
        cost = F.leaky_relu(cost, 0.1,inplace=True)
        return cost

    def corr(self, refimg_fea, targetimg_fea, maxdisp):
        device = refimg_fea.device
        b,c,h,w = refimg_fea.shape

        if refimg_fea.is_cuda:
            refimg_fea_span = torch.cuda.FloatTensor(2*maxdisp+1,2*maxdisp+1,b,c,h,w).to(device)
        else:
            refimg_fea_span = torch.FloatTensor(2*maxdisp+1,2*maxdisp+1,b,c,h,w)

        for i in range(2*maxdisp+1):
            ind = i-maxdisp
            for j in range(2*maxdisp+1):
                iind = j-maxdisp
                if ind < 0:
                    srci_st, srci_ed, tari_st, tari_ed = -ind, w, 0, w + ind
                else:
                    srci_st, srci_ed, tari_st, tari_ed = 0, w - ind, ind, w
                if iind < 0:
                    srcj_st, srcj_ed, tarj_st, tarj_ed = -iind, h, 0, h + iind
                else:
                    srcj_st, srcj_ed, tarj_st, tarj_ed = 0, h - iind, iind, h
                refimg_fea_span[i, j, :, :, tari_st: tari_ed, tarj_st: tarj_ed] = refimg_fea[:, :, srci_st: srci_ed, srcj_st: srcj_ed]

        cost = refimg_fea_span * targetimg_fea
        cost = F.leaky_relu(cost, 0.1,inplace=True)
        return cost

    def get_oor_loss(self, flowl0, oor3, maxdisp, occ_mask):
        """
        return out-of-range loss
        """
        oor3_gt = (flowl0.abs() > maxdisp).detach() #  (8*self.md[3])
        oor3_gt = (((oor3_gt.sum(1)>0) + occ_mask)>0).float()  # oor, or occluded
        weights = oor3_gt.sum().float()/(oor3_gt.shape[0]*oor3_gt.shape[1]*oor3_gt.shape[2])
        weights = oor3_gt * (1-weights) + (1-oor3_gt) * weights
        loss_oor3 = F.binary_cross_entropy_with_logits(oor3,oor3_gt,size_average=True, weight=weights)
        return loss_oor3

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    #@profile
    def forward(self,im_A,im_B,disc_aux=None):
        bs = im_A.shape[0]
        # for alignment
        im = torch.cat((im_A, im_B), dim=0)

        c06,c05,c04,c03,c02 = self.pspnet(im)
        # 64, 32, 16, 8, 4
        c16 = c06[:bs];  c26 = c06[bs:]
        c15 = c05[:bs];  c25 = c05[bs:]
        c14 = c04[:bs];  c24 = c04[bs:]
        c13 = c03[:bs];  c23 = c03[bs:]
        c12 = c02[:bs];  c22 = c02[bs:]

        # normalize
        c16n = c16 / (c16.norm(dim=1, keepdim=True)+1e-9)
        c26n = c26 / (c26.norm(dim=1, keepdim=True)+1e-9)
        c15n = c15 / (c15.norm(dim=1, keepdim=True)+1e-9)
        c25n = c25 / (c25.norm(dim=1, keepdim=True)+1e-9)
        c14n = c14 / (c14.norm(dim=1, keepdim=True)+1e-9)
        c24n = c24 / (c24.norm(dim=1, keepdim=True)+1e-9)
        c13n = c13 / (c13.norm(dim=1, keepdim=True)+1e-9)
        c23n = c23 / (c23.norm(dim=1, keepdim=True)+1e-9)
        c12n = c12 / (c12.norm(dim=1, keepdim=True)+1e-9)
        c22n = c22 / (c22.norm(dim=1, keepdim=True)+1e-9)

        ## matching 6
        feat6 = self.corr(c16n,c26n,self.md[0])
        feat6 = self.f6(feat6)
        cost6 = self.p6(feat6) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost6.shape
        cost6 = cost6.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow6h,ent6h = self.flow_reg64(cost6) # bx16, 2, h, w
        flow6h =  flow6h.view(bs,-1,h,w) # b, 16*2, h, w
        ent6h =  ent6h.view(bs,-1,h,w) # b, 16*1, h, w


        # hypotheses fusion
        x = torch.cat((ent6h.detach(), flow6h.detach(), c16),1)
        x = self.dc6_conv4(self.dc6_conv3(self.dc6_conv2(self.dc6_conv1(x))))
        va = self.dc6_conv7(self.dc6_conv6(self.dc6_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow6 = ( flow6h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)

        ## matching 5
        up_flow6 = F.upsample(flow6, [im.size()[2]//32,im.size()[3]//32], mode='bilinear')*2
        warp5,_ = self.warp5(c25n, up_flow6)
        feat5 = self.corr(c15n,warp5,self.md[1])
        feat5 = self.f5(feat5)
        cost5 = self.p5(feat5) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost5.shape
        cost5 = cost5.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow5h,ent5h = self.flow_reg32(cost5) # bx16, 2, h, w
        flow5h = flow5h.view(b,c,2,h,w) + up_flow6[:,np.newaxis]
        flow5h = flow5h.view(bs,-1,h,w) # b, 16*2, h, w
        ent5h =  ent5h.view(bs,-1,h,w) # b, 16*1, h, w

        # append coarse hypotheses
        flow5h = torch.cat((flow5h, F.upsample(flow6h.detach()*2, [flow5h.shape[2],flow5h.shape[3]], mode='bilinear')),1) # b, k2--k2, h, w
        ent5h = torch.cat((ent5h, F.upsample(ent6h, [flow5h.shape[2],flow5h.shape[3]], mode='bilinear')),1)


        # hypotheses fusion
        x = torch.cat((ent5h.detach(), flow5h.detach(), c15),1)
        x = self.dc5_conv4(self.dc5_conv3(self.dc5_conv2(self.dc5_conv1(x))))
        va5 = self.dc5_conv7(self.dc5_conv6(self.dc5_conv5(x)))
        va5 = va5.view(b,-1,2,h,w)
        flow5 = ( flow5h.view(b,-1,2,h,w) * F.softmax(va5,1) ).sum(1) # b, 2k, 2, h, w


        ## matching 4
        up_flow5 = F.upsample(flow5, [im.size()[2]//16,im.size()[3]//16], mode='bilinear')*2
        warp4,_ = self.warp4(c24n, up_flow5)

        feat4 = self.corr(c14n,warp4,self.md[2])
        feat4 = self.f4(feat4)
        cost4 = self.p4(feat4) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost4.shape
        cost4 = cost4.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow4h,ent4h = self.flow_reg16(cost4) # bx16, 2, h, w
        flow4h = flow4h.view(b,c,2,h,w) + up_flow5[:,np.newaxis]
        flow4h =  flow4h.view(bs,-1,h,w) # b, 16*2, h, w
        ent4h =  ent4h.view(bs,-1,h,w) # b, 16*1, h, w

        # append coarse hypotheses
        flow4h = torch.cat((flow4h, F.upsample(flow5h.detach()*2, [flow4h.shape[2],flow4h.shape[3]], mode='bilinear')),1)
        ent4h = torch.cat((ent4h, F.upsample(ent5h, [flow4h.shape[2],flow4h.shape[3]], mode='bilinear')),1)


        # hypotheses fusion
        x = torch.cat((ent4h.detach(), flow4h.detach(), c14),1)
        x =  self.dc4_conv4(self.dc4_conv3(self.dc4_conv2(self.dc4_conv1(x))))
        va = self.dc4_conv7(self.dc4_conv6(self.dc4_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow4 = ( flow4h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)


        ## matching 3
        up_flow4 = F.upsample(flow4, [im.size()[2]//8,im.size()[3]//8], mode='bilinear')*2
        warp3,_ = self.warp3(c23n, up_flow4)

        feat3 = self.corr(c13n,warp3,self.md[3])
        feat3 = self.f3(feat3)
        cost3 = self.p3(feat3) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost3.shape
        cost3 = cost3.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow3h,ent3h = self.flow_reg8(cost3) # bx16, 2, h, w
        flow3h = flow3h.view(b,c,2,h,w) + up_flow4[:,np.newaxis]
        flow3h = flow3h.view(bs,-1,h,w) # b, 16*2, h, w
        ent3h =  ent3h.view(bs,-1,h,w) # b, 16*1, h, w

        # append coarse hypotheses
        flow3h = torch.cat((flow3h, F.upsample(flow4h.detach()*2, [flow3h.shape[2],flow3h.shape[3]], mode='bilinear')),1)
        ent3h = torch.cat((ent3h, F.upsample(ent4h, [flow3h.shape[2],flow3h.shape[3]], mode='bilinear')),1)


        # hypotheses fusion
        x = torch.cat((ent3h.detach(), flow3h.detach(), c13),1)
        x = self.dc3_conv4(self.dc3_conv3(self.dc3_conv2(self.dc3_conv1(x))))
        va = self.dc3_conv7(self.dc3_conv6(self.dc3_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow3 = ( flow3h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)

        ## matching 2
        up_flow3 = F.upsample(flow3, [im.size()[2]//4,im.size()[3]//4], mode='bilinear')*2
        warp2,_ = self.warp2(c22n, up_flow3)
        feat2 = self.corr(c12n,warp2,self.md[4])
        feat2 = self.f2(feat2)
        cost2 = self.p2(feat2) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost2.shape
        cost2 = cost2.view(-1,u,v,h,w)  # bx12, 9,9,h,w, also predict uncertainty from here
        flow2h,ent2h = self.flow_reg4(cost2) # bx12, 2, h, w
        flow2h = flow2h.view(b,c,2,h,w) + up_flow3[:,np.newaxis]
        flow2h = flow2h.view(bs,-1,h,w) # b, 12*2, h, w
        ent2h =  ent2h.view(bs,-1,h,w) # b, 12*1, h, w

        # append coarse hypotheses
        flow2h = torch.cat((flow2h, F.upsample(flow3h.detach()*2, [flow2h.shape[2],flow2h.shape[3]], mode='bilinear')),1)
        ent2h = torch.cat((ent2h, F.upsample(ent3h, [flow2h.shape[2],flow2h.shape[3]], mode='bilinear')),1)

        # hypotheses fusion
        x = torch.cat((ent2h.detach(), flow2h.detach(), c12),1)
        x = self.dc2_conv4(self.dc2_conv3(self.dc2_conv2(self.dc2_conv1(x))))
        va = self.dc2_conv7(self.dc2_conv6(self.dc2_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow2 = ( flow2h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)


        flow2 = F.upsample(flow2, [im.size()[2],im.size()[3]], mode='bilinear')
        flow3 = F.upsample(flow3, [im.size()[2],im.size()[3]], mode='bilinear')
        flow4 = F.upsample(flow4, [im.size()[2],im.size()[3]], mode='bilinear')
        flow5 = F.upsample(flow5, [im.size()[2],im.size()[3]], mode='bilinear')
        flow6 = F.upsample(flow6, [im.size()[2],im.size()[3]], mode='bilinear')


        return flow2*4, ent2h.mean(1)[0]
