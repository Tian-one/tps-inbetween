
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
# from _util.util_v0 import * ; import _util.util_v0 as uutil
# from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
# from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch
import imageio
import os
import numpy as np
from PIL import Image
import torch
from util import utils


class ML240(Dataset):
    def __init__(self, data_path, size=(512 ,512), xN=2, gray=True, match_dir=None, pre_process=False, train=False):
        self.data_path = data_path
        self.folder_path = [os.path.join(data_path, folder) for folder in sorted(os.listdir(data_path))]#[0:1]

        self.folder_name = sorted(os.listdir(data_path))
        self.xN = xN
        self.size = size
        self.gray = gray
        self.all_frame_path = []
        self.all_frame_name = []
        self.match_dir = match_dir
        self.pre_porcess = pre_process
        self.train = train
        
        self.all_matches = []
        self.all_frame = []

        h, w = self.size[0], self.size[1]
        c = 1 if self.gray else 3

        for idx, folder in enumerate(self.folder_path):
            f_name = sorted(os.listdir(folder))
            frame = [os.path.join(folder, f) for f in f_name]
            self.all_frame_path.append(frame)
            if self.pre_porcess:
                self.all_frame.append([self.img_open_torch(imgs_pth, self.size, self.gray) for imgs_pth in frame])
                if idx % 10 == 0:
                    print('loading ', idx, ' finish!')
            b_name = [(self.folder_name[idx] + '-' + n) for n in f_name]
            self.all_frame_name.append(b_name)

        
        if self.match_dir:
            self.match_folder_path = [os.path.join(self.match_dir, folder) for folder in sorted(os.listdir(self.match_dir))]#[0:1]
            for idx, folder in enumerate(self.match_folder_path):
                f_name = sorted(os.listdir(folder))
                frame = [os.path.join(folder, f) for f in f_name if os.path.isfile(os.path.join(folder, f))]
                self.all_matches.append(frame)
                assert len(frame) + self.xN == len(self.all_frame_path[idx])
            assert len(self.all_frame_path) == len(self.all_matches)
        


    def __len__(self):
        length = 0
        for v in self.all_frame_path:
            length += (len(v) - self.xN)
        return length

    def get_video_idx(self, idx):
        tmp_idx = idx
        for v_idx, v in enumerate(self.all_frame_path):
            if tmp_idx >= (len(v) - self.xN):
                tmp_idx = tmp_idx - (len(v) - self.xN)
            else:
                return v_idx, tmp_idx
        return v_idx, tmp_idx

    def img_open_torch(self, img_pth, size, gray):
        img = Image.open(img_pth).resize(size)
        if gray:
            img = img.convert('L')
            np_img = np.array(img)[..., None]
        else:
            np_img = np.array(img)
        torch_img = np2Tensor(np_img)
        return torch_img
    

    def __getitem__(self, idx):
        v_idx, f_idx = self.get_video_idx(idx)
        if self.pre_porcess:
            imgs = self.all_frame[v_idx][f_idx:f_idx +self.xN +1]
        else:
            imgs_pth = self.all_frame_path[v_idx][f_idx:f_idx +self.xN +1]
            imgs = [self.img_open_torch(_pth, self.size, self.gray) for _pth in imgs_pth]

        basename = self.all_frame_name[v_idx][f_idx:f_idx +self.xN +1]


        img_input = [imgs[0], imgs[self.xN]]
        img_gt = imgs[1:self.xN]


        if self.match_dir:
            matched_path = self.all_matches[v_idx][f_idx]    
            return img_input, img_gt, basename, matched_path
        
        return img_input, img_gt, basename


def get_loader(data_path, batch_size, shuffle, match_dir=None, img_size=(512, 512), num_workers=0, xN=6, train=False, gray=True, pre_process=False):
    dataset = ML240(data_path, xN=xN, size=img_size, gray=gray, match_dir=match_dir, pre_process=pre_process, train=train)
    if not train:
        dataset = Subset(dataset, range(0, 1000))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_val_loader(data_path, batch_size, shuffle, match_dir=None, img_size=(512, 512), num_workers=0, xN=6, train=False, gray=True, pre_process=False):
    dataset = ML240(data_path, xN=xN, size=img_size, gray=gray, match_dir=match_dir, pre_process=pre_process)
    indices = range(len(dataset) // 4)
    val_dataset = Subset(dataset, indices)

    return DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def np2Tensor(img, rgb_range=1, n_colors=1):
    img = img.astype('float64')
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # NHWC -> NCHW
    tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
    tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
    return tensor


