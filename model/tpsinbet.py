import torch
import torch.nn as nn
import torch.nn.functional as F
import util.utils as utils
import torchvision
import numpy as np
from model.warplayer import warp
from model.IFNet import IFNet


class TPS_inbet(nn.Module):
    def __init__(self, args):
        super(TPS_inbet, self).__init__()

        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.times = torch.linspace(0, 1, args.xN+1)[1:-1].to(self.device)
        self.encoder = Contextnet(indim=1)
        self.IFNet = IFNet()
        self.args = args
        # for k, v in pretrained_model.items():
        #     k = k[7:]
        #     new_state_dict[k] = v        
        # self.IFNet.load_state_dict(new_state_dict)
        self.unet = Unet(indim=9, outdim=1)


    def warp(self, img, flow, size):
        flow_ = flow.reshape((size[0], size[2], size[3], 2))  # b,2,h,w - b,h,w,2
        grid = utils.grid_from_flow(flow_, size)
        warped = F.grid_sample(img, grid, padding_mode='border')
        return warped

    def warp_zero(self, img, flow, size):
        flow_ = flow.reshape((size[0], size[2], size[3], 2))  # b,2,h,w - b,h,w,2
        grid = utils.grid_from_flow(flow_, size)
        warped = F.grid_sample(img, grid)
        return warped
    
    def forward(self, x0, x1, matches_path=None, middle=False, aug=False):

        b, c, h, w = x0.shape 
        # grid01 = []
        # grid10 = []
        flow01 = []
        flow10 = []
        out = []
        coarseI_m = []

        with torch.no_grad():
            if matches_path is not None: 
                # tps coarse motion estimate
                for i in range(b):
                    kps = np.load(matches_path[i]).astype(np.float32)
                    if kps.shape[1] <= 2:
                        flow01_ = torch.zeros((1, 2, h, w), dtype=torch.float32).to(self.device)
                        flow10_ = torch.zeros((1, 2, h, w), dtype=torch.float32).to(self.device)
                    else:
                        kps0 = torch.from_numpy(kps[0]).to(self.device)
                        kps1 = torch.from_numpy(kps[1]).to(self.device)
                        theta10_ = utils.tps_theta_from_points(kps1, kps0, reduced=True)
                        theta01_ = utils.tps_theta_from_points(kps0, kps1, reduced=True)
                        grid01_, flow01_ = utils.tps_grid(theta01_.unsqueeze(0), kps1.unsqueeze(0), (1, c, h, w))
                        grid10_, flow10_ = utils.tps_grid(theta10_.unsqueeze(0), kps0.unsqueeze(0), (1, c, h, w))

                        flow01_ = flow01_.view((1,2,h,w))
                        flow10_ = flow10_.view((1,2,h,w))

                    flow01.append(flow01_)
                    flow10.append(flow10_)
            else:
                # without tps motion estimate
                for i in range(b):
                    flow01_ = torch.zeros((1, 2, h, w), dtype=torch.float32).to(self.device)
                    flow10_ = torch.zeros((1, 2, h, w), dtype=torch.float32).to(self.device)
                    flow01.append(flow01_)
                    flow10.append(flow10_)


        coarse_flow01 = torch.cat(flow01, dim=0)
        coarse_flow10 = torch.cat(flow10, dim=0)
        
        # motion refine
        for idx, t in enumerate(self.times):
            # assume linear motion
            coarse_flow0t = coarse_flow01 * t * t - (1-t)*t*coarse_flow10
            coarse_flow1t = coarse_flow10 * (1-t) * (1-t) - t*(1-t)*coarse_flow01

            # apply backward warp
            coarse_I0t = self.warp(x0, coarse_flow0t, (b,c,h,w))
            coarse_I1t = self.warp(x1, coarse_flow1t, (b,c,h,w))

            # fine motion estimation
            feats0 = self.encoder(x0, coarse_flow0t)
            feats1 = self.encoder(x1, coarse_flow1t)
            flow_bet_coar, m, interp, _, _, _ = self.IFNet(coarse_I0t, coarse_I1t)

            fine_I0t = warp(coarse_I0t, flow_bet_coar[:, :2])
            fine_I1t = warp(coarse_I1t, flow_bet_coar[:, 2:4])

            # synthesize frames
            refine_out = self.unet(x0, x1, coarse_flow0t+flow_bet_coar[:, :2], coarse_flow1t+flow_bet_coar[:, 2:4], fine_I0t, fine_I1t, m, feats0, feats1)
            res = refine_out[:, 0:1, :, :]*2 - 1
            It =  (1-t)*fine_I0t + t*fine_I1t + res

            It = torch.clamp(It, 0, 1)
            coarseI_m.append(coarse_I0t)
            out.append(It)

        return out, coarseI_m


def make_model(args):
    return TPS_inbet(args)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Contextnet(nn.Module):
    def __init__(self, indim=1, c=16):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(indim, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


class Unet(nn.Module):
    def __init__(self, indim=8, outdim=1, c=16):
        super(Unet, self).__init__()
        self.down0 = Conv2(indim, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, outdim, 3, 1, 1)

    def forward(self, img0, img1, flow0, flow1, warped_img0, warped_img1, mask, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, flow0, flow1, mask), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        x = self.conv(x)
        return torch.sigmoid(x)