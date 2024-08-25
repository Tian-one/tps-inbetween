import argparse
import os
from os.path import join
import kornia
import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from model.gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from model.gluestick.models.two_view_pipeline import TwoViewPipeline
# from kornia.geometry.ransac import RANSAC
from kornia.feature.adalam.adalam import AdalamFilter
from skimage.measure import ransac
from skimage.transform import AffineTransform
import torch.nn.functional as F
from PIL import Image
import time
from datasets.ml240 import get_loader
import tqdm

def flow_vis(flow, radius=0.2):
    # print(flow)
    assert len(flow.shape)==3 and flow.shape[0]==2
    h,w = flow.shape[-2:]
    m = max(h,w)
    r = radius * m
    hsv = torch.stack([
        torch.atan2(flow[0], flow[1]) + np.pi,
        (torch.norm(flow, dim=0) / r).clip(0, 1),
        torch.ones(h, w, device=flow.device),
    ])
    rgb = kornia.color.hsv_to_rgb(hsv)
    # print(rgb)
    return rgb

def tensor2numpy(tensor, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = np.transpose(img.cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8)
    return img

def norm_flow(flow, h, w):
    # flow_norm = flow.clone()
    flow_norm = flow.copy()
    flow_norm[:, 0] = flow_norm[:, 0] / w
    flow_norm[:, 1] = flow_norm[:, 1] / h
    return flow_norm


def main():
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default=(720, 720))
    parser.add_argument('--data_path', default='/hdd/zty/datasets/Anime/ml240data/ml100_norm/all/frames')
    parser.add_argument('--max_pts', type=int, default=1000)
    parser.add_argument('--max_lines', type=int, default=300)
    parser.add_argument('--xN', type=int, default=6)
    parser.add_argument('--save_path', default='./match_example')
    args = parser.parse_args()

    dataloader = get_loader(data_path=args.data_path, batch_size=1, shuffle=False, img_size=args.size, xN=args.xN)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Evaluation config
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': args.max_pts,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': args.max_lines,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline_model = TwoViewPipeline(conf).to(device).eval()
    
    # ransac = RANSAC(max_iter=50)
    for idx, (img_input, img_gt, basename) in tqdm.tqdm(enumerate(dataloader)):
        torch_gray0 = img_input[0].to(device)
        torch_gray1 = img_input[1].to(device)
        folder_name = basename[0][0].split('-')[0]
        b, c, h, w = torch_gray0.shape
        save_folder = os.path.join(args.save_path, folder_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_match_name = os.path.join(save_folder, str(idx) + '.npy')

        x = {'image0': torch_gray0, 'image1': torch_gray1}
        s_t = time.time()
        pred = pipeline_model(x)
        m_t = time.time() - s_t
        print(m_t)
        pred = batch_to_np(pred)

        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0 = pred["matches0"]

        # line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
        line_matches = pred["line_matches0"]

        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]


        n_kps0 = norm_flow(matched_kps0, h, w)
        n_kps1 = norm_flow(matched_kps1, h, w)

        kps_stack = np.stack((n_kps0, n_kps1), axis=0)
        print(kps_stack.shape)
        np.save(save_match_name, kps_stack)


    

if __name__ == '__main__':

    main()
