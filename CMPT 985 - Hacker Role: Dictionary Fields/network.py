import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Notes:
# - Bilinear interpolation is implemented in torch.nn.functional.grid_sample and is not explictly defined here
# -

# The following functions are directly taken from the original implementation of the dictionary field
# Only edits made are deletions of unused functions and minor edits to variable names
# Original available here: https://github.com/autonomousvision/factor-fields/

def get_coeff(self, xyz_sampled):
    N_points, dim = xyz_sampled.shape
    in_dim = self.in_dim
    pts = self.normalize_coord(xyz_sampled).view([1, -1] + [1] * (dim - 1) + [dim])
    coeffs = F.grid_sample(self.coeffs[self.scene_idx], pts, mode=self.cfg.model.coef_mode, align_corners=False,
                               padding_mode='border').view(-1, N_points).t()

def grid_mapping(positions, freq_bands, aabb, basis_mapping='sawtooth'):
    aabbSize = max(aabb[1] - aabb[0])
    scale = aabbSize[..., None] / freq_bands
    if basis_mapping == 'triangle':
        pts_local = (positions - aabb[0]).unsqueeze(-1) % scale
        pts_local_int = ((positions - aabb[0]).unsqueeze(-1) // scale) % 2
        pts_local = pts_local / (scale / 2) - 1
        pts_local = torch.where(pts_local_int == 1, -pts_local, pts_local)
    elif basis_mapping == 'sawtooth':
        pts_local = (positions - aabb[0])[..., None] % scale
        pts_local = pts_local / (scale / 2) - 1
        pts_local = pts_local.clamp(-1., 1.)
    elif basis_mapping == 'sinc':
        pts_local = torch.sin((positions - aabb[0])[..., None] / (scale / np.pi) - np.pi / 2)
    elif basis_mapping == 'trigonometric':
        pts_local = (positions - aabb[0])[..., None] / scale * 2 * np.pi
        pts_local = torch.cat((torch.sin(pts_local), torch.cos(pts_local)), dim=-1)
    elif basis_mapping == 'x':
        pts_local = (positions - aabb[0]).unsqueeze(-1) / scale
    # elif basis_mapping=='hash':
    #     pts_local = (positions - aabb[0])/max(aabbSize)

def N_to_reso(n_voxels, aabb):
    xyz_min, xyz_max = aabb
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min) / n_voxels).pow(1 / dim)
    return torch.round((xyz_max - xyz_min) / voxel_size).long().tolist()

def dct_dict(n_atoms_fre, size, n_select, dim=2):
    """
    Create a dictionary using the Discrete Cosine Transform (DCT) basis. If n_atoms is
    not a perfect square, the returned dictionary will have ceil(sqrt(n_atoms))**2 atoms
    :param n_atoms:
        Number of atoms in dict
    :param size:
        Size of first patch dim
    :return:
        DCT dictionary, shape (size*size, ceil(sqrt(n_atoms))**2)
    """
    # todo flip arguments to match random_dictionary
    p = n_atoms_fre  # int(math.ceil(math.sqrt(n_atoms)))
    dct = np.zeros((p, size))

    for k in range(p):
        basis = np.cos(np.arange(size) * k * math.pi / p)
        if k > 0:
            basis = basis - np.mean(basis)

        dct[k] = basis

    kron = np.kron(dct, dct)
    if 3 == dim:
        kron = np.kron(kron, dct)

    if n_select < kron.shape[0]:
        idx = [x[0] for x in np.array_split(np.arange(kron.shape[0]), n_select)]
        kron = kron[idx]

    for col in range(kron.shape[0]):
        norm = np.linalg.norm(kron[col]) or 1
        kron[col] /= norm

    kron = torch.FloatTensor(kron)
    return kron

# The following functions & classes encompass my "hacker role" implementation of Dictionary Fields
# I chose to operate on 256x256 slices of MRI volumes, simplifying to a 2D regression task
# I experiment with sharing basis between registered slices in various imaging modalities {T1w, T2w, SWI} \
# against a baseline of learning a separate basis for each slice

def create_dense_grid(grid_single_dim_len, res, mask):
    grid_1dim = np.linspace(start=0, stop=grid_single_dim_len, num=res, endpoint=True)
    x, y, z = grid_1dim, grid_1dim, grid_1dim
    grid = np.meshgrid(x, y, z, sparse=False, indexing='xy')
    if mask:
        grid = np.where(mask, grid, 0)
    return grid


def grid_mapping_hacker(positions, freq_bands, aabb):
    aabbSize = max(aabb[1] - aabb[0])
    scale = aabbSize[..., None] / freq_bands
     # basis_mapping == 'sawtooth':
    pts_local = (positions - aabb[0])[..., None] % scale
    pts_local = pts_local / (scale / 2) - 1
    pts_local = pts_local.clamp(-1., 1.)
    return pts_local


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_layers=2, with_dropout=False):
        super(SimpleMLP, self).__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_layers
        self.with_dropout = with_dropout

        backbone = []
        for l in range(num_layers):
            if l == 0:
                layer_in_dim = self.in_dim
            else:
                layer_in_dim = self.hidden_dim

            if l == num_layers:
                layer_out_dim, bias = out_dim, False
            else:
                layer_out_dim, bias = self.hidden_dim, True

            backbone.append(nn.Linear(layer_in_dim, layer_out_dim, bias=bias))
        self.backbone = nn.ModuleList(backbone)

        def forward(self, x, is_train=False):
            y_hat = x
            if self.with_dropout and self.is_train:
                y_hat = F.dropout(x, p=0.1)
            for l in range(self.num_layers):
                y_hat = self.backbone[l](x)
                if l != self.num_layers - 1:
                    x = F.relu(x, inplace=True)

            return y_hat


class DictionaryField_2DRegression(pl.LightningModule):
    def __init__(self, **kwargs):
        super(DictionaryField_2DRegression, self).__init__()
        self.kwargs = kwargs
        self.HW = np.array([256, 256])
        self.in_dim = 2
        self.out_dim = 1
        self.basis_dims = np.array([32, 32, 32, 16, 16, 16])
        self.basis_resos = np.linspace(32, 256, 6)
        self.coef_reso = 32
        self.coef_init = 0.001
        self.hidden_dim = 64
        self.with_dropout = False
        self.loss_scale = 1.0
        self.training_step_cnt = 0
        self.total_params = 1426063
        self.aabb = torch.FloatTensor([[0., 0.], [256., 256.]])[:, :self.in_dim]
        self.total_basis = sum(np.power(np.array(self.basis_resos), self.in_dim) * np.array(self.basis_dims))
        self.total_coeff = self.total_params - self.total_basis
        self.frequency_bands = max(self.aabb[1][:self.in_dim]) / torch.FloatTensor(self.basis_resos)
        self.coeff_reso = N_to_reso(self.total_coeff // sum(self.basis_dims), self.aabb[:, :self.in_dim])[::-1]
        self.coeff_reso = [self.aabb[1][-1]] + self.coeff_reso
        self.n_scene = 1
        self.small_mlp = SimpleMLP(in_dim=2, out_dim=1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.002)

    def training_step(self, batch, batch_idx):
        coordinates, density = batch
        feats, coeff = self.get_coding(coordinates)
        y_hat = self.small_mlp(feats, is_train=True)
        loss = torch.mean((y_hat.squeeze() - density) ** 2)
        psnr = -10.0 * np.log(loss.item()) / np.log(10.0)

        loss = loss * self.loss_scale
        if self.training_step_cnt % 100 == 0:
            key = f'{self.training_step_cnt= }'
            self.log_image(key=key, images=y_hat.squeeze())

    def normalize_coord(self, xyz):
        inv_aabb_size = 2.0 / (self.aabb[1] - self.aabb[0])
        return (xyz - self.aabb[0]) * inv_aabb_size - 1.0

    def init_coeff(self):
        n_scene = self.n_scene
        coeffs = [self.coef_init * torch.ones((1, sum(self.basis_dims), *self.coeff_reso)) for _ in range(n_scene)]
        return torch.nn.ParameterList(coeffs)

    def init_basis(self):
        basises, coeffs, n_params_basis = [], [], 0
        for i, (basis_dim, reso) in enumerate(zip(self.basis_dims, self.basis_reso)):
            basises.append(torch.nn.Parameter(dct_dict(int(np.power(basis_dim, 1. / self.in_dim) + 1),
                                                       reso, n_select=basis_dim,
                                                       dim=self.in_dim).reshape([1, basis_dim] + [reso] * self.in_dim)))
        return torch.nn.ParameterList(basises)

    def get_basis(self, x):
        n_points = x.shape[0]
        x = x[..., :-1]
        num_freq = len(self.freq_bands)
        xyz = grid_mapping_hacker(x, self.freq_bands, self.aabb[:, :self.in_dim]).view(
            1, *([1] * (self.in_dim - 1)), -1, self.in_dim, num_freq)
        basises = []
        for i in range(num_freq):
            basises.append(F.grid_sample(self.basises[i], xyz[..., i], mode='bilinear', align_corners=True).view(
                -1, n_points).T)
        if isinstance(basises, list):
            basises = torch.cat(basises, dim=-1)
        return basises

    def get_coeffs(self, xyz_sampled):
        N_points, dim = xyz_sampled.shape
        in_dim = self.in_dim
        pts = self.normalize_coord(xyz_sampled).view([1, -1] + [1] * (dim - 1) + [dim])
        coeffs = F.grid_sample(self.coeffs[self.scene_idx], pts, mode=self.cfg.model.coef_mode, align_corners=False,
                               padding_mode='border').view(-1, N_points).t()
        return coeffs

    def get_coding(self, x):
        coeff = self.get_coeffs(x)
        basises = self.get_basis(x)
        return basises * coeff, coeff