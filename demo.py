import argparse
import os
import kornia
import cv2
import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from model.gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from model.gluestick.models.two_view_pipeline import TwoViewPipeline
import torch.nn.functional as F
from PIL import Image
import copy
from datasets.ml240 import get_loader
from model.tpsinbet import TPS_inbet
from util.utils import batch_dog
import imageio

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

def np2Tensor(img, rgb_range=1, n_colors=1):
    img = img.astype('float64')
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # NHWC -> NCHW
    tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
    tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
    return tensor

def norm_flow(flow, h, w):
    flow_norm = flow.copy()
    flow_norm[:, 0] = flow_norm[:, 0] / w
    flow_norm[:, 1] = flow_norm[:, 1] / h
    return flow_norm


def img_open_torch(img_pth, size=None, gray=True):
    img = Image.open(img_pth)
    if size:
        img = img.resize(size)
    if gray:
        img = img.convert('L')
        np_img = np.array(img)[..., None]
    else:
        np_img = np.array(img)
    torch_img = np2Tensor(np_img)
    return torch_img.unsqueeze(0)

def binarization(img, th=0.95):
    img_copy = copy.deepcopy(img)
    img_copy[img_copy<th] = 0
    img_copy[img_copy>=th] = 1
    return img_copy

def img2gif(imgs_pth, save_pth, dur=0.1):
    frames = []
    for file_name in imgs_pth:
        frames.append(imageio.v2.imread(file_name))
    imageio.mimsave(save_pth, frames, duration=dur)

def main():
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', default='assets/input1_0.png')
    parser.add_argument('--image2', default='assets/input1_1.png')
    parser.add_argument('--xN', type=int, default=30)
    parser.add_argument('--save_path', default='./output')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--model_path', type=str, default='ckpt/model_latest.pt')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Matching model config
    print("======= Start Loading Matching Model======")
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': 1000,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': 300,
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

    matching_model = TwoViewPipeline(conf).to(device).eval()

    print("Successfully loading matching model! Start matching ...")
        
    torch_gray0 = img_open_torch(args.image1).to(device)
    torch_gray1 = img_open_torch(args.image2).to(device)

    b, c, h, w = torch_gray0.shape
    x = {'image0': torch_gray0, 'image1': torch_gray1}
    pred = matching_model(x)
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



    matching_image_path = os.path.join(args.save_path, 'matching.png')
    matching_np_path = os.path.join(args.save_path, 'matching.npy')
     # Plot the matches
    img0, img1 = cv2.cvtColor(torch_gray0.squeeze(0).squeeze(0).cpu().numpy(), cv2.COLOR_RGB2BGR), cv2.cvtColor(torch_gray1.squeeze(0).squeeze(0).cpu().numpy(), cv2.COLOR_RGB2BGR)
    img0, img1 = np.uint8(img0*255), np.uint8(img1*255)

    image1 = cv2.applyColorMap(img0, cv2.COLORMAP_HOT)
    image2 = cv2.applyColorMap(img1, cv2.COLORMAP_OCEAN)

    overlap = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    for i in range(len(matched_kps0)):
        # print((int(matched_kps0[i, 0]), int(matched_kps0[i, 1])), (int(matched_kps1[i, 0]), int(matched_kps1[i, 1])))
        cv2.line(overlap, (int(matched_kps0[i, 0]), int(matched_kps0[i, 1])), (int(matched_kps1[i, 0]), int(matched_kps1[i, 1])),
                        (0, 255, 0), 1)

    cv2.imwrite(matching_image_path, overlap)

    kps_stack = np.stack((n_kps0, n_kps1), axis=0)
    np.save(matching_np_path, kps_stack)

    print("Matching complete! Results have saved to ", matching_image_path)
    print("====== Start Inbetweening ======")
    # model = tps_only(args)
    model = TPS_inbet(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device).eval()
    with torch.no_grad():
        pred, _ = model(1-torch_gray0, 1-torch_gray1, [matching_np_path])

    pred = [1-p for p in pred]

    save_images_path = os.path.join(args.save_path, 'images')
    if not os.path.exists(save_images_path):
        os.mkdir(save_images_path)

    torchvision.transforms.ToPILImage()(torch_gray0.float().squeeze(0)).save(os.path.join(save_images_path, '0.png'))
    torchvision.transforms.ToPILImage()(torch_gray1.float().squeeze(0)).save(os.path.join(save_images_path, str(len(pred)+1)+'.png'))
    for idx, img_pred in enumerate(pred):
        torchvision.transforms.ToPILImage()(img_pred.float().squeeze(0)).save(os.path.join(save_images_path, str(idx+1)+'.png'))
    img_pth = [os.path.join(save_images_path, '{}.png'.format(i)) for i in range(args.xN+1)]
    img2gif(img_pth, os.path.join(args.save_path, 'out.gif'), dur=0.05)

    print("Finish! Results have saved to ", os.path.join(args.save_path, 'out.gif'))

if __name__ == '__main__':
    main()
