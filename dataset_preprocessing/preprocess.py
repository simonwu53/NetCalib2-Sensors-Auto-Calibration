import argparse
import random
import cv2
import numpy as np
from numpy.random import default_rng
from typing import Tuple, Literal, Dict, Union, Optional, List, Callable
from pathlib import Path
import pykitti
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm
import time

# seeds
SEED = 53
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
RNG = default_rng(SEED)


def angle_to_rotation_matrix(rot):
    """
    Transform vector of Euler angles to rotation matrix
    ref: http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html

    :param rot: euler angle PyTorch Tensor (length 3 shape (3,), roll-pitch-yaw or x-y-z)
    :return: 3*3 rotation matrix
    """
    s_u, s_v, s_w = np.sin(rot)
    c_u, c_v, c_w = np.cos(rot)

    # keep tracking the gradients, devices, and dtype
    return np.array([[c_v*c_w, s_u*s_v*c_w-c_u*s_w, s_u*s_w+c_u*s_v*c_w],
                    [c_v*s_w, c_u*c_w+s_u*s_v*s_w, c_u*s_v*s_w-s_u*c_w],
                    [-s_v, s_u*c_v, c_u*c_v]], dtype=np.float32)


def phi_to_transformation_matrix(phi):
    """
    Transform calibration vector to calibration matrix (Numpy version)
    \theta_{calib} = [r_x,r_y,r_z,t_x,t_y,t_z]^T -> \phi_{calib} 4*4 matrix

    :param phi: calibration PyTorch Tensor (length 6, shape (6,)), which is an output from calibration network

    :return: transformation matrix from Lidar coordinates to camera's frame
    """
    # split rotation & translation values
    rot, trans = phi[:3], phi[3:]
    # get rotation matrix
    rot_mat = angle_to_rotation_matrix(rot)

    # create transformation matrix
    T = np.zeros((4, 4), dtype=np.float32)

    T[:3, :3] = rot_mat
    T[:3, 3] = trans
    T[3, 3] = 1
    return T


def crop_img(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """bottom-center cropping (crop the sky part and keep center)"""
    assert len(img.shape) >= 2, 'img must be a shape of (H, W) or (H, W, C)'
    height, width = img.shape[0], img.shape[1]
    top_margin = int(height - h)
    left_margin = int((width - w) / 2)

    if len(img.shape) == 3:
        image = img[top_margin:top_margin + h, left_margin:left_margin + w, :]
    else:
        image = img[top_margin:top_margin + h, left_margin:left_margin + w]
    return image


def lidar_projection(scan: np.ndarray, T: np.ndarray, P: np.ndarray, shape: Union[np.ndarray, Tuple[int, int]],
                     R: Optional[np.ndarray] = None, crop: Optional[Union[np.ndarray, Tuple[int, int]]] = None) -> np.ndarray:
    # Reflectance > 0
    pts3d = scan[scan[:, 3] > 0, :]
    pts3d[:, 3] = 1

    # coordinates transformation
    if R is None:
        pts3d_cam = T @ pts3d.transpose()
    else:
        pts3d_cam = R @ (T @ pts3d.transpose())

    # Before projecting, keep only points with z>0
    # (points that are in front of the camera).
    idx = pts3d_cam[2, :] > 0
    pts2d_cam = P @ pts3d_cam[:, idx]

    # get projected 2d & 3d points
    pts3d = pts3d[idx]
    pts2d = pts2d_cam / pts2d_cam[2, :]

    # keep points projected in the image plane
    pts2d = pts2d.transpose().round().astype(np.int32)
    xmin, ymin = 0, 0
    xmax, ymax = shape[1], shape[0]
    mask = (xmin < pts2d[:, 0]) & (pts2d[:, 0] < xmax) & \
           (ymin < pts2d[:, 1]) & (pts2d[:, 1] < ymax)
    pts2d = pts2d[mask][:, :2]  # keep only coordinates
    pts3d = pts3d[mask][:, 0]  # keep only x values of scan (depth)

    # draw depth map
    depth = np.zeros(shape, dtype=np.float32)
    depth[pts2d[:, 1], pts2d[:, 0]] = pts3d

    # crop
    if crop is not None:
        depth = crop_img(depth, crop[0], crop[1])
    return depth


def sync_file_names(files1: List[str], files2: List[str]) -> Tuple[List[str], List[str]]:
    idx_f1 = np.array([int(f.split('/')[-1].split('.')[0]) for f in files1])  # index from files1
    idx_f2 = np.array([int(f.split('/')[-1].split('.')[0]) for f in files2])  # index from files2
    if idx_f2.shape[0] < idx_f1.shape[0]:
        files1 = [files1[i] for i, idx in enumerate(idx_f1) if idx in idx_f2]
    else:
        files2 = [files2[i] for i, idx in enumerate(idx_f2) if idx in idx_f1]

    new_idx_f1 = np.array([int(f.split('/')[-1].split('.')[0]) for f in files1])
    new_idx_f2 = np.array([int(f.split('/')[-1].split('.')[0]) for f in files2])
    assert np.isin(new_idx_f1, new_idx_f2).all(), 'Error occurred when syncing files...'
    return files1, files2


class ErrorGen:
    def __init__(self, rot_off: float, trans_off: float):
        assert rot_off > 0 and trans_off > 0, 'rotation and translation offsets must be greater than 0, and symmetric interval'
        self.trans_off = trans_off                                                                 # unit: meters
        self.rot_off = np.deg2rad(rot_off)                                                         # unit: degrees

        self.dist = RNG.uniform                                                                    # uniform generator
        self.rot_mean, self.trans_mean = 0, 0                                                      # mean of uniform distribution
        self.rot_std, self.trans_std = 2*self.rot_off/np.sqrt(12), 2*self.trans_off/np.sqrt(12)    # std of uniform distribution
        return

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.dist(-self.rot_off, self.rot_off, 3), self.dist(-self.trans_off, self.trans_off, 3)

    def standardize(self, rot: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (rot-self.rot_mean)/self.rot_std, (trans-self.trans_mean)/self.trans_std

    def destandardize(self, rot: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return rot*self.rot_std+self.rot_mean, trans*self.trans_std+self.trans_mean


class KITTIDataset(Dataset):
    def __init__(self, mode: Literal['train', 'test', 'val'] = 'train',
                 database: str = '/home/shanwu/dataset/KITTI/',
                 selected_date: str = '2011_09_26',
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 crop_size: Tuple[int, int] = (352, 1216),
                 merge_drives: bool = True):

        super(KITTIDataset, self).__init__()
        self.merge_drives = merge_drives
        self.loader = None
        self.normalize = Normalize(mean=mean, std=std)
        self.crop_h, self.crop_w = crop_size
        self.current_kitti_loader = None
        self.current_kitti_len = 0

        # get dataset base
        mode = mode.lower()
        assert mode in ('train', 'test', 'val'), "mode should be one of the 'train', 'test', 'val'."
        self.kitti_base = Path(database).joinpath(mode)
        self.date = selected_date
        self.pred_save_path = self.kitti_base.joinpath(self.date)
        # get drive numbers in dataset
        self.drives = [i.name.split('_')[4] for i in self.pred_save_path.glob(f'{self.date}_drive_*_sync')]

        # data dict
        self.loaders = {}
        self.kitti_loaders = {}
        self.total_len = 0
        for drive in self.drives:
            loader = pykitti.raw(self.kitti_base.as_posix(), self.date, drive)
            self.kitti_loaders[drive] = loader
            drive_len = len(loader.cam2_files)
            self.loaders[range(self.total_len, self.total_len+drive_len)] = loader
            self.total_len += drive_len
        return

    def __getitem__(self, idx: int) -> Dict:
        if self.merge_drives:
            for k in self.loaders:
                if idx in k:
                    self.loader = self.loaders[k]
                    idx = idx = idx-k.start
        else:
            assert self.current_kitti_loader is not None, "No kitti loader found."
            self.loader = self.current_kitti_loader

        focal = float(self.loader.calib.P_rect_20[0][0])

        # get img & crop to (352, 1216)
        raw2 = (np.asarray(self.loader.get_cam2(idx), np.float32) / 255.0)
        raw3 = (np.asarray(self.loader.get_cam3(idx), np.float32) / 255.0)
        # height, width = raw.shape[0], raw.shape[1]
        # top_margin = int(height - self.crop_h)
        # left_margin = int((width - self.crop_w) / 2)
        # image = raw[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
        image2 = crop_img(raw2, self.crop_h, self.crop_w)
        image3 = crop_img(raw3, self.crop_h, self.crop_w)

        # convert to tensor, shape (C, H, W) & normalize
        image2 = self.normalize(torch.from_numpy(image2.transpose((2, 0, 1))))
        image3 = self.normalize(torch.from_numpy(image3.transpose((2, 0, 1))))

        # velodyne
        point_cloud = self.loader.get_velo(idx)

        # camera params
        T = self.loader.calib.T_cam2_velo.astype(np.float32)
        P = self.loader.calib.P_rect_20.astype(np.float32)

        return {'image2': image2, 'image3': image3, 'focal': focal, 'raw': raw2,
                'velo': point_cloud, 'shape': np.array(raw2.shape[:2]), 'P': P, 'T': T}

    def __len__(self) -> int:
        if self.merge_drives:
            return self.total_len
        else:
            return self.current_kitti_len

    def set_kitti_loader(self, drive: str) -> None:
        assert drive in self.drives, f'drive number: {drive} not found.'
        self.current_kitti_loader = self.kitti_loaders[drive]
        self.current_kitti_len = len(self.current_kitti_loader.cam2_files)
        return

    def get_drive_path(self, drive: str) -> Path:
        assert drive in self.drives, f'drive number: {drive} not found.'
        return self.pred_save_path.joinpath(f'2011_09_26_drive_{drive}_sync')

    def get_drive_data(self, drive: str, channels: Literal['rgb', 'bgr'] = 'bgr') -> np.ndarray:
        assert drive in self.drives, f'drive number: {drive} not found.'
        self.loader = pykitti.raw(self.kitti_base.as_posix(), self.date, drive)
        if channels == 'bgr':
            return np.concatenate([cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)[np.newaxis, :] for img in self.loader.cam2])
        else:
            return np.concatenate([np.array(img, dtype=np.uint8)[np.newaxis, :] for img in self.loader.cam2])


def preprocess_depth_data(dataset: KITTIDataset, method_for_cam_depth: Optional, rot_off: float, trans_off: float) -> None:
    print(f"Pre-processing KITTI depth dataset, total drives: {len(dataset.drives)}, total images: {dataset.total_len}")
    t0 = time.time()
    err_generator = ErrorGen(rot_off, trans_off)

    # cv2.namedWindow('pred', flags=cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('pred', 1250, 400)
    # cv2.namedWindow('proj', flags=cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('proj', 1250, 400)

    for i, drive in enumerate(dataset.drives):
        # set kitti loader
        dataset.set_kitti_loader(drive)

        # mkdir
        drive_path = dataset.pred_save_path.joinpath(f'{dataset.date}_drive_{drive}_sync')
        image_c_path = drive_path.joinpath('image_c')
        image_l_path = drive_path.joinpath('image_l')

        if image_c_path.exists() and image_l_path.exists():
            print(f'Drive {drive} seems already processed. Skip.')
            continue
        if len(dataset.current_kitti_loader.cam2_files) != len(dataset.current_kitti_loader.velo_files):
            print(f'Drive {drive} has mismatch cam2 files and velo files! Fixing...')
            kitti_loader = dataset.current_kitti_loader
            # syncing data
            kitti_loader.cam2_files, kitti_loader.velo_files = sync_file_names(kitti_loader.cam2_files, kitti_loader.velo_files)
            # reset dataset length
            dataset.set_kitti_loader(drive)

        image_c_path.mkdir(exist_ok=True)
        image_l_path.mkdir(exist_ok=True)
        f_gt_err = open(drive_path.joinpath('gt.txt'), 'w')

        # data loader and progress bar
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers, pin_memory=True, drop_last=False)
        pbar = tqdm(enumerate(loader))
        pbar.set_description(f'[Drive {drive} ({i + 1}/{len(dataset.drives)})]')

        # creating depth maps
        with torch.no_grad():
            for j, sample in pbar:
                # get data
                img2 = sample['image2'].cuda()
                img3 = sample['image3'].cuda()
                f = sample['focal'].float().cuda()
                velo = sample['velo'].numpy()[0]
                shape = sample['shape'].numpy()[0]
                P = sample['P'].numpy()[0]
                T = sample['T'].numpy()[0]

                # get lidar depth
                gt_err = np.concatenate(err_generator.sample(), dtype=np.float32)
                T_err = phi_to_transformation_matrix(gt_err)
                T_composed = T_err @ T
                depth_l = lidar_projection(velo, T_composed, P, shape, crop=(dataset.crop_h, dataset.crop_w))
                depth_l = (depth_l*256).astype(np.uint16)

                # get cam2 depth
                if method_for_cam_depth is not None:
                    depth_c = (method_for_cam_depth(img2, img3) * 256.0).astype(np.uint16)

                # write to disk
                if method_for_cam_depth is not None:
                    cv2.imwrite(image_c_path.joinpath(f'{j:010d}.png').as_posix(), depth_c, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(image_l_path.joinpath(f'{j:010d}.png').as_posix(), depth_l, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                f_gt_err.write(f'{j:010d}.png,' + ','.join(map(str, gt_err)) + '\n')

                # cv2.imshow('pred', depth_c)
                # cv2.imshow('proj', (depth_l*256).astype(np.uint16))
                # key = cv2.waitKey(int(1 / 60 * 1000))
                # if (key & 0xFF == ord('q')) or (key & 0xFF == 27):
                #     break

        print(f'Drive {drive} done. time: {(time.time()-t0)/60:.2f} min.')
        f_gt_err.close()

    return


def play_sequence(data: np.ndarray, win_size: Tuple[int] = (1250, 400), rate: int = 10) -> None:
    assert len(data.shape) == 4, 'Data must be a ndarray of shape (N, H, W, C).'
    cv2.namedWindow('player', flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('player', win_size[0], win_size[1])
    for i in range(data.shape[0]):
        cv2.imshow('player', data[i])
        key = cv2.waitKey(int(1/rate*1000))
        if (key & 0xFF == ord('q')) or (key & 0xFF == 27):
            break
    cv2.destroyAllWindows()
    return


def arg_parser():
    parser = argparse.ArgumentParser(description='Script for KITTI dataset pre-processing.')
    parser.add_argument('--data_path', type=str, help='path to the data', required=True)
    parser.add_argument('--n_workers', type=int, help='number of workers for data loader', default=6)
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], help='train test val split', default='train')
    parser.add_argument('--rotation_offset', type=float, help='rotation offset (degrees) used to create un-calibrated dataset', default=2)
    parser.add_argument('--translation_offset', type=float, help='translation offset (meters) used to create un-calibrated dataset', default=0.2)
    parser.add_argument('--no_cam_depth', action='store_true', help='Only pre-processing the LiDAR projection maps.')
    return parser.parse_args()


if __name__ == '__main__':
    # parse args
    args = arg_parser()
    print(args)

    # initialize methods for camera depth here
    func_for_cam_depth: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    if not args.no_cam_depth:
        # ------------
        # TODO: if you want to generate camera depth maps, implement here
        # This function will be called at line #312.
        # Inputs are tensors, you can convert it back to numpy array if your algorithm don't use tensors
        #
        # typing: Callable[[Tensor, Tensor], Tensor]
        # input : img2, Tensor, left color image from KITTI dataset, loaded by PyTorch DataLoader
        #         img3, Tensor, right color image from KITTI dataset, loaded by PyTorch DataLoader
        # output: depth_c, Tensor, camera depth map, value range [0, 1]
        # ------------
        # func_for_cam_depth: Callable = SomeMethodOfYourChoice()
        ...
        assert func_for_cam_depth is not None, \
            'You need to add your algorithm above for generating camera depth maps. ' \
            'It can be as simple as a Semi-Global Block Matching (SGBM) algorithm or' \
            'a much complicated neural network based algorithm.'

    # load dataset
    dataset = KITTIDataset(mode=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers, pin_memory=True, drop_last=True)

    # tasks
    dataset.merge_drives = False
    preprocess_depth_data(dataset, func_for_cam_depth, args.rotation_offset, args.translation_offset)
