from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch
import numpy as np
from pathlib import Path
from typing import Literal, Tuple, Union, Optional
from tqdm import tqdm
import pykitti2
from utils import read_depth


# statistics
M_VELOD = 14.0833
S_VELOD = 8.7353
M_CAM2D = 17.2444
S_CAM2D = 14.5541


def run_stat(loader: DataLoader):
    # statistics
    n_samples = len(loader.dataset)
    n_pixels_per_frame = np.prod((352, 1216))
    x_sum_c = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # cam depth sum of x
    x2_sum_c = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # cam depth sum of x^2
    x_sum_l = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # lidar depth sum of x
    x2_sum_l = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # lidar depth sum of x^2
    n_pixels_total_c = torch.tensor(n_samples * n_pixels_per_frame, dtype=torch.float32,
                                    device='cuda', requires_grad=False)  # total pixels
    n_pixels_total_l = 0

    # iterating over dataset
    with torch.no_grad():
        for i, sample in enumerate(tqdm(loader)):
            # unpack data
            cam2_d, velo_d, gt_err, gt_err_norm = sample[0].cuda(), sample[1].cuda(), sample[2], sample[3].cuda()

            n_pixels_total_l += velo_d[0][velo_d[0] != 0].shape[0]

            # track the sum of x and the sum of x^2
            x_sum_l += torch.sum(velo_d[0])
            x2_sum_l += torch.sum(velo_d[0] ** 2)
            x_sum_c += torch.sum(cam2_d[0])
            x2_sum_c += torch.sum(cam2_d[0] ** 2)

    # calculate mean and std.
    # formula: stddev = sqrt((SUM[x^2] - SUM[x]^2 / n) / (n-1))
    mean_l = x_sum_l / n_pixels_total_l
    mean_c = x_sum_c / n_pixels_total_c
    std_l = torch.sqrt((x2_sum_l - x_sum_l ** 2 / n_pixels_total_l) / (n_pixels_total_l - 1))
    std_c = torch.sqrt((x2_sum_c - x_sum_c ** 2 / n_pixels_total_c) / (n_pixels_total_c - 1))
    print('Lidar Depth Map Statistics')
    print('Mean: %.4f' % mean_l.cpu().item())
    print('STD: %.4f' % std_l.cpu().item())
    print()
    print('Camera Depth Map Statistics')
    print('Mean: %.4f' % mean_c.cpu().item())
    print('STD: %.4f' % std_c.cpu().item())
    return


class TrainDataset(Dataset):
    def __init__(self, path: str, date: str = '2011_09_26', mode: Literal['train', 'val', 'test'] = 'train',
                 rotation_offset: float = 2.0, translation_offset: float = 0.2):
        super(TrainDataset, self).__init__()
        assert len(date.split('_')) == 3, "Date must has the format of 'YYYY_MM_DD'."
        self.date = date
        self.mode = None
        self.drives = None
        self.total_len = None
        self.loaders = None

        self.rotation_offset = np.deg2rad(rotation_offset)
        self.translation_offset = translation_offset
        self.rot_mean = 0
        self.trans_mean = 0
        self.rot_std = 2*self.rotation_offset/np.sqrt(12)
        self.trans_std = 2*self.translation_offset/np.sqrt(12)

        self.root = Path(path)
        self.train_root = self.root.joinpath('train')
        self.val_root = self.root.joinpath('val')
        self.test_root = self.root.joinpath('test')

        self.set_mode(mode)
        return

    def set_mode(self, mode: Literal['train', 'val', 'test'] = 'train'):
        self.drives = [i.name.split('_')[4] for i in getattr(self, f'{mode}_root').joinpath(self.date).glob(f'{self.date}_drive_*_sync')]
        self.mode = mode
        self._init_data()
        return

    def destandardize(self, rot: Union[Tensor, np.ndarray], trans: Union[Tensor, np.ndarray]) -> \
            Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
        return rot*self.rot_std+self.rot_mean, trans*self.trans_std+self.trans_mean

    def _init_data(self):
        self.total_len = 0
        self.loaders = {}
        for drive in self.drives:
            loader = pykitti2.raw(getattr(self, f'{self.mode}_root').as_posix(), self.date, drive)
            gt_file = getattr(self, f'{self.mode}_root').joinpath(self.date).joinpath(f'{self.date}_drive_{drive}_sync').joinpath('gt.txt').as_posix()
            with open(gt_file, 'r') as f:
                gt = [l.strip().split(',') for l in f]
            drive_len = loader.get_depth_len()
            self.loaders[range(self.total_len, self.total_len+drive_len)] = (loader, gt)
            self.total_len += drive_len
        return

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        velo_batch = np.zeros((130000, 4), dtype=np.float32)
        # find idx in which data loader
        for k in self.loaders:
            if idx in k:
                # load data
                loader, gt = self.loaders[k]
                # get real idx
                idx = idx - k.start

                # load depth maps & normalize
                cam2_d = (read_depth(loader.get_cam2d(idx))[np.newaxis, :, :] - M_CAM2D) / S_CAM2D
                velo_d = (read_depth(loader.get_velod(idx))[np.newaxis, :, :] - M_VELOD) / S_VELOD

                # calc errors
                gt_err = np.array(gt[idx][1:], dtype=np.float32)
                gt_err_norm = gt_err.copy()
                gt_err_norm[:3] = (gt_err_norm[:3]-self.rot_mean)/self.rot_std
                gt_err_norm[3:] = (gt_err_norm[3:]-self.trans_mean)/self.trans_std

                # lidar pc
                velo = loader.get_velo(idx)
                velo_batch[:velo.shape[0]] = velo
                # cam2 = np.array(loader.get_cam2d(idx), dtype=np.uint8)
                T = loader.calib.T_cam2_velo.astype(np.float32)

                return cam2_d, velo_d, gt_err, gt_err_norm, velo_batch, T
        raise IndexError(f'Could not find data by index: {idx}')

    def __len__(self) -> int:
        return self.total_len


class EvalDataset(Dataset):
    def __init__(self, path: str, date: str = '2011_09_26', mode: Literal['train', 'val', 'test'] = 'train',
                 rotation_offset: float = 2.0, translation_offset: float = 0.2):
        super(EvalDataset, self).__init__()
        assert len(date.split('_')) == 3, "Date must has the format of 'YYYY_MM_DD'."
        self.date = date
        self.mode = None
        self.drives = None
        self.total_len = None
        self.loaders = None

        self.rotation_offset = np.deg2rad(rotation_offset)
        self.translation_offset = translation_offset
        self.rot_mean = 0
        self.trans_mean = 0
        self.rot_std = 2*self.rotation_offset/np.sqrt(12)
        self.trans_std = 2*self.translation_offset/np.sqrt(12)

        self.root = Path(path)
        self.train_root = self.root.joinpath('train')
        self.val_root = self.root.joinpath('val')
        self.test_root = self.root.joinpath('test')

        self.set_mode(mode)
        return

    def set_mode(self, mode: Literal['train', 'val', 'test'] = 'train'):
        self.drives = [i.name.split('_')[4] for i in getattr(self, f'{mode}_root').joinpath(self.date).glob(f'{self.date}_drive_*_sync')]
        self.mode = mode
        self._init_data()
        return

    def destandardize(self, rot: Union[Tensor, np.ndarray], trans: Union[Tensor, np.ndarray],
                      rot_offset: Optional[float] = None, trans_offset: Optional[float] = None) -> \
            Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
        if rot_offset is None:
            return rot*self.rot_std+self.rot_mean, trans*self.trans_std+self.trans_mean
        else:
            rot_offset = np.deg2rad(rot_offset)
            trans_offset = trans_offset
            rot_mean = 0
            trans_mean = 0
            rot_std = 2 * rot_offset / np.sqrt(12)
            trans_std = 2 * trans_offset / np.sqrt(12)
            return rot*rot_std+rot_mean, trans*trans_std+trans_mean

    def _init_data(self):
        self.total_len = 0
        self.loaders = {}
        for drive in self.drives:
            loader = pykitti2.raw(getattr(self, f'{self.mode}_root').as_posix(), self.date, drive)
            gt_file = getattr(self, f'{self.mode}_root').joinpath(self.date).joinpath(f'{self.date}_drive_{drive}_sync').joinpath('gt.txt').as_posix()
            with open(gt_file, 'r') as f:
                gt = [l.strip().split(',') for l in f]
            drive_len = loader.get_depth_len()
            self.loaders[range(self.total_len, self.total_len+drive_len)] = (loader, gt)
            self.total_len += drive_len
        return

    def __getitem__(self, idx) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        # find idx in which data loader
        for k in self.loaders:
            if idx in k:
                # load data
                loader, gt = self.loaders[k]
                # get real idx
                idx = idx - k.start

                # load depth maps & normalize
                cam2_d = (read_depth(loader.get_cam2d(idx))[np.newaxis, :, :] - M_CAM2D) / S_CAM2D
                velo_d = (read_depth(loader.get_velod(idx))[np.newaxis, :, :] - M_VELOD) / S_VELOD

                # calc errors
                gt_err = np.array(gt[idx][1:], dtype=np.float32)
                gt_err_norm = gt_err.copy()
                gt_err_norm[:3] = (gt_err_norm[:3]-self.rot_mean)/self.rot_std
                gt_err_norm[3:] = (gt_err_norm[3:]-self.trans_mean)/self.trans_std

                if self.mode == 'test':
                    velo = loader.get_velo(idx)
                    cam2 = np.array(loader.get_cam2(idx), dtype=np.uint8)
                    T = loader.calib.T_cam2_velo.astype(np.float32)
                    P = loader.calib.P_rect_20.astype(np.float32)
                    return cam2_d, velo_d, gt_err, gt_err_norm, cam2, velo, T, P
                else:
                    return cam2_d, velo_d, gt_err, gt_err_norm
        raise IndexError(f'Could not find data by index: {idx}')

    def __len__(self) -> int:
        return self.total_len
