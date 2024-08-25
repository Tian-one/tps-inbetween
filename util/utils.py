# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import numpy as np
import torch
import math

import torch.nn as nn
import kornia
from kornia.filters import filter2d
from kornia.utils import create_meshgrid
from scipy.ndimage import distance_transform_edt as distance

class TPS:
    @staticmethod
    def fit(c, lambd=0., reduced=False):
        n = c.shape[0]
        device = c.device

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, dtype=torch.float32, device=device) * lambd

        P = torch.ones((n, 3), dtype=torch.float32, device=device)
        P[:, 1:] = c[:, :2]

        v = torch.zeros(n + 3, dtype=torch.float32, device=device)
        v[:n] = c[:, -1]

        A = torch.zeros((n + 3, n + 3), dtype=torch.float32, device=device)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        theta = torch.linalg.solve(A, v)  # p has structure w, a
        return theta[1:] if reduced else theta

    @staticmethod
    def d(a, b):
        return torch.sqrt(torch.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r**2 * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        x = torch.atleast_2d(x)
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = torch.cat((-torch.sum(w, keepdim=True), w))
        b = torch.matmul(U, w)
        return a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + b

def uniform_grid(shape):
    '''Uniform grid coordinates.
    
    Params
    ------
    shape : tuple
        HxW defining the number of height and width dimension of the grid

    Returns
    -------
    points: HxWx2 tensor
        Grid coordinates over [0,1] normalized image range.
    '''

    H,W = shape[:2]    
    c = np.empty((H, W, 2))
    c[..., 0] = np.linspace(0, 1, W, dtype=np.float32)
    c[..., 1] = np.expand_dims(np.linspace(0, 1, H, dtype=np.float32), -1)

    return c
    
def tps_theta_from_points(c_src, c_dst, reduced=False):
    delta = c_src - c_dst

    cx = torch.cat((c_dst,delta[:, 0].view(-1, 1)), dim=1)
    cy = torch.cat((c_dst, delta[:, 1].view(-1, 1)), dim=1)

    theta_dx = TPS.fit(cx, reduced=reduced)
    theta_dy = TPS.fit(cy, reduced=reduced)

    return torch.stack((theta_dx, theta_dy), dim=-1)



def tps_theta_from_points_torch(c_src, c_dst, reduced=False):
    b, _, _ = c_src.shape
    device = c_src.device
    l_theta = []
    for i in range(b):
        theta = tps_theta_from_points(c_src[i].numpy(), c_dst[i].numpy(), reduced=reduced)
        l_theta.append(torch.from_numpy(theta))

    return torch.stack(l_theta, dim=0)


def tps_grid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.
    
    Params
    ------
    grid : HxWx2 array
        Normalized flow field coordinates as computed by compute_densegrid.
    sshape : tuple
        Height and width of source image in pixels.


    Returns
    -------
    mapx : HxW array
    mapy : HxW array
    '''

    mx = (grid[:, :, 0] * sshape[1]).astype(np.float32)
    my = (grid[:, :, 1] * sshape[0]).astype(np.float32)

    return mx, my

def tps(theta, ctrl, grid):
    '''Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
    The TPS surface is a minimum bend interpolation surface defined by a set of control points.
    The function value for a x,y location is given by
    
        TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])
        
    This method computes the TPS value for multiple batches over multiple grid locations for 2 
    surfaces in one go.
    
    Params
    ------
    theta: Nx(T+3)x2 tensor, or Nx(T+2)x2 tensor
        Batch size N, T+3 or T+2 (reduced form) model parameters for T control points in dx and dy.
    ctrl: NxTx2 tensor or Tx2 tensor
        T control points in normalized image coordinates [0..1]
    grid: NxHxWx3 tensor
        Grid locations to evaluate with homogeneous 1 in first coordinate.
        
    Returns
    -------
    z: NxHxWx2 tensor
        Function values at each grid location in dx and dy.
    '''
    
    N, H, W, _ = grid.size()

    if ctrl.dim() == 2:
        ctrl = ctrl.expand(N, *ctrl.size())
    
    T = ctrl.shape[1]
    
    diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)
    D = torch.sqrt((diff**2).sum(-1))
    U = (D**2) * torch.log(D + 1e-6)

    w, a = theta[:, :-3, :], theta[:, -3:, :]

    reduced = T + 2  == theta.shape[1]
    if reduced:
        w = torch.cat((-w.sum(dim=1, keepdim=True), w), dim=1) 

    # U is NxHxWxT
    b = torch.bmm(U.view(N, -1, T), w).view(N,H,W,2)
    # b is NxHxWx2
    z = torch.bmm(grid.view(N,-1,3), a).view(N,H,W,2) + b
    
    return z

def tps_grid(theta, ctrl, size):
    '''Compute a thin-plate-spline grid from parameters for sampling.
    
    Params
    ------
    theta: Nx(T+3)x2 tensor
        Batch size N, T+3 model parameters for T control points in dx and dy.
    ctrl: NxTx2 tensor, or Tx2 tensor
        T control points in normalized image coordinates [0..1]
    size: tuple
        Output grid size as NxCxHxW. C unused. This defines the output image
        size when sampling.
    
    Returns
    -------
    grid : NxHxWx2 tensor
        Grid suitable for sampling in pytorch containing source image
        locations for each output pixel.
    '''    
    N, _, H, W = size

    grid = theta.new(N, H, W, 3)
    grid[:, :, :, 0] = 1.
    grid[:, :, :, 1] = torch.linspace(0, 1, W)
    grid[:, :, :, 2] = torch.linspace(0, 1, H).unsqueeze(-1)   
    
    z = tps(theta, ctrl, grid)
    return (grid[...,1:] + z)*2-1, z # [-1,1] range required by F.sample_grid

def grid_from_flow(flow, size):

    N, _, H, W = size
    grid = flow.new(N, H, W, 3)
    grid[:, :, :, 0] = 1.
    grid[:, :, :, 1] = torch.linspace(0, 1, W)
    grid[:, :, :, 2] = torch.linspace(0, 1, H).unsqueeze(-1)  

    return (grid[...,1:] + flow)*2-1


def tps_sparse(theta, ctrl, xy):
    if xy.dim() == 2:
        xy = xy.expand(theta.shape[0], *xy.size())

    N, M = xy.shape[:2]
    grid = xy.new(N, M, 3)
    grid[..., 0] = 1.
    grid[..., 1:] = xy

    z = tps(theta, ctrl, grid.view(N,M,1,3))
    return xy + z.view(N, M, 2)

def uniform_grid(shape):
    '''Uniformly places control points aranged in grid accross normalized image coordinates.
    
    Params
    ------
    shape : tuple
        HxW defining the number of control points in height and width dimension

    Returns
    -------
    points: HxWx2 tensor
        Control points over [0,1] normalized image range.
    '''
    H,W = shape[:2]    
    c = torch.zeros(H, W, 2)
    c[..., 0] = torch.linspace(0, 1, W)
    c[..., 1] = torch.linspace(0, 1, H).unsqueeze(-1)
    return c


def norm_match_torch(flow, h, w):
    '''
    match shape: (B, N, 2)
    '''
    flow_norm = flow.clone()
    flow_norm[:, :, 0] = flow_norm[:, :, 0] / w
    flow_norm[:, :, 1] = flow_norm[:, :, 1] / h
    return flow_norm


def distance_transform(image: torch.Tensor, kernel_size: int = 3, h: float = 0.35) -> torch.Tensor:
    r"""Approximates the Manhattan distance transform of images using cascaded convolution operations.

    The value at each pixel in the output represents the distance to the nearest non-zero pixel in the image image.
    It uses the method described in :cite:`pham2021dtlayer`.
    The transformation is applied independently across the channel dimension of the images.

    Args:
        image: Image with shape :math:`(B,C,H,W)`.
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    Returns:
        tensor with shape :math:`(B,C,H,W)`.

    Example:
        >>> tensor = torch.zeros(1, 1, 5, 5)
        >>> tensor[:,:, 1, 2] = 1
        >>> dt = kornia.contrib.distance_transform(tensor)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image type is not a torch.Tensor. Got {type(image)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # n_iters is set such that the DT will be able to propagate from any corner of the image to its far,
    # diagonally opposite corner
    n_iters: int = math.ceil(max(image.shape[2], image.shape[3]) / math.floor(kernel_size / 2))
    grid = create_meshgrid(
        kernel_size, kernel_size, normalized_coordinates=False, device=image.device, dtype=image.dtype
    )

    grid -= math.floor(kernel_size / 2)
    kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
    kernel = torch.exp(kernel / -h).unsqueeze(0)

    out = torch.zeros_like(image)

    # It is possible to avoid cloning the image if boundary = image, but this would require modifying the image tensor.
    boundary = image.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(n_iters):
        cdt = filter2d(boundary, kernel, border_type='replicate')
        cdt = -h * torch.log(cdt + 1e-5)

        # We are calculating log(0) above.
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        mask = torch.where(cdt > 0, 1.0, 0.0)
        if mask.sum() == 0:
            break

        offset: int = i * kernel_size // 2
        out += (offset + cdt) * mask
        boundary = torch.where(mask == 1, signal_ones, boundary)

    return out


def distance_transform_norm(image: torch.Tensor, kernel_size: int = 3, h: float = 0.35, norm_factor: float=15):
    h, w = image.shape[-2], image.shape[-1]
    out = distance_transform(image, kernel_size, h)
    # out = 1 - (-out / norm_factor).exp()
    out = out / np.sqrt(h**2+w**2)
    return out

def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    fg_dtm = 1 - (-fg_dtm / 15).exp()
    return fg_dtm


def batch_dog(img, t=1.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=True):
    # to grayscale if needed
    # print("111", img.shape)
    bs,ch,h,w = img.shape
    if ch in [3,4]:
        img = kornia.color.rgb_to_grayscale(img[:,:3])
    else:
        assert ch==1

    # calculate dog
    kern0 = max(2*int(sigma*kernel_factor)+1, 3)
    kern1 = max(2*int(sigma*k*kernel_factor)+1, 3)
    g0 = kornia.filters.gaussian_blur2d(
        img, (kern0,kern0), (sigma,sigma), border_type='replicate',
    )
    g1 = kornia.filters.gaussian_blur2d(
        img, (kern1,kern1), (sigma*k,sigma*k), border_type='replicate',
    )
    ans = 0.5 + t*(g1-g0) - epsilon
    ans = ans.clip(0,1) if clip else ans
    return ans
