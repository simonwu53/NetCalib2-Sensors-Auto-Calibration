import random
import numpy as np
import argparse
import cv2
import time
import torch
from torch import Tensor
import torch.utils.data
from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional, List
from net.CalibrationNet import get_model
from dataset import EvalDataset as CalibDataset
from utils import count_parameters, inv_transform_vectorized, phi_to_transformation_matrix_vectorized, \
    lidar_projection, inv_transform, phi_to_transformation_matrix, merge_color_img_with_depth

# set seeds
SEED = 53
rng = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda'


class Loss(Module):
    def __init__(self, dataset: CalibDataset, reduction: str = 'mean',
                 alpha: float = 0.5, beta: float = 1, gamma: float = 0.5, cache: int = 60000):
        super(Loss, self).__init__()
        self.dataset = dataset
        self.cache = cache

        self.mse_loss_fn_r = MSELoss(reduction=reduction)
        self.mse_loss_fn_t = MSELoss(reduction=reduction)
        self.center_loss_fn = MSELoss(reduction=reduction)
        self.depth_loss_fn = MSELoss(reduction=reduction)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        return

    def forward(self, pred: Tuple[Tensor, Tensor], target: Tuple[Tensor, Tensor],
                velo: Tensor, T: Tensor):
        rot_pred_, trans_pred_ = self.dataset.destandardize(*pred)
        rot_err_, trans_err_ = self.dataset.destandardize(*target)

        T_err = phi_to_transformation_matrix_vectorized(rot_err_, trans_err_, device=DEVICE)
        T_fix = inv_transform_vectorized(phi_to_transformation_matrix_vectorized(rot_pred_, trans_pred_, device=DEVICE), device=DEVICE)
        T_recalib = T_fix.bmm(T_err.bmm(T))

        # reproject velo points
        pts3d_cam_l = []
        pts3d_cam_recalib_l = []
        for i in range(velo.shape[0]):
            scan = velo[i]
            T_ = T[i]
            T_recalib_ = T_recalib[i]
            # Reflectance > 0
            pts3d = scan[scan[:, 3] > 0, :]
            pts3d[:, 3] = 1
            # project
            pts3d_cam = T_.mm(pts3d.t()).t()[:self.cache, :].unsqueeze(0)
            pts3d_cam_recalib = T_recalib_.mm(pts3d.t()).t()[:self.cache, :].unsqueeze(0)
            pts3d_cam_l.append(pts3d_cam)
            pts3d_cam_recalib_l.append(pts3d_cam_recalib)
        pts3d_cam = torch.cat(pts3d_cam_l, dim=0)
        pts3d_cam_recalib = torch.cat(pts3d_cam_recalib_l, dim=0)

        loss_mse = self.mse_loss(pred, target)
        loss_center = self.center_loss(pts3d_cam_recalib, pts3d_cam)
        loss_depth = self.depth_loss(pts3d_cam_recalib, pts3d_cam)
        return loss_mse + loss_center + loss_depth

    def mse_loss(self, pred: Tuple[Tensor, Tensor], target: Tuple[Tensor, Tensor]):
        rot, trans = pred
        gt_r, gt_t = target
        loss_r = self.mse_loss_fn_r(rot, gt_r)
        loss_t = self.mse_loss_fn_t(trans, gt_t)
        return self.alpha * (loss_r + loss_t)

    def center_loss(self, pts_recalib: Tensor, pts_orig: Tensor):
        c_orig = pts_orig[:, :, :3].mean(dim=1)
        c_recalib = pts_recalib[:, :, :3].mean(dim=1)
        return self.beta * self.center_loss_fn(c_recalib, c_orig)

    def depth_loss(self, pts_recalib: Tensor, pts_orig: Tensor):
        return self.gamma * self.depth_loss_fn(pts_recalib, pts_orig)


def run_visualization(model: Module, loader: DataLoader, loss_fn: Loss, fout: Optional[str] = None, downscale: int = 2) -> None:
    h, w = 375//downscale, 1242//downscale
    # if fout is not None:
    #     pout = Path(fout).joinpath('model_inference.avi')
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     writer = cv2.VideoWriter(pout.as_posix(), fourcc, 10.0, (1242, 375*3))

    cv2.namedWindow('Visualization', flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Visualization', w, h*3)

    running_loss = 0
    running_mean = 0
    running_time = 0
    running_msee = 0
    rot_err = np.zeros((len(loader.dataset), 6), np.float32)
    trans_err = np.zeros((len(loader.dataset), 6), np.float32)
    pbar = tqdm(loader)
    model.eval()

    with torch.no_grad():
        i = 0
        for i, sample in enumerate(pbar):
            cam2_d, velo_d, gt_err, gt_err_norm, cam2, velo, T, P = sample[0].cuda(), sample[1].cuda(), sample[2], sample[3].cuda(), \
                                                              sample[4].cuda(), sample[5].cuda(), sample[6].cuda(), sample[7].cuda()
            t = time.time()
            pred_r, pred_t = model(velo_d, cam2_d)
            running_time += (time.time() - t)
            loss = loss_fn((pred_r, pred_t), torch.split(gt_err_norm, 3, dim=1), velo, T)

            # collect statistics
            running_loss += loss.item()
            running_mean = running_loss / (i + 1)
            pbar.set_description("Valid running loss: %.4f" % (running_mean))

            gt_splits = np.split(gt_err.numpy(), 2, axis=1)
            pred_r, pred_t = loader.dataset.destandardize(pred_r.detach().cpu().numpy(), pred_t.detach().cpu().numpy())
            rot_err[i, :3] = np.abs(pred_r-gt_splits[0]).mean(axis=0)
            trans_err[i, :3] = np.abs(pred_t-gt_splits[1]).mean(axis=0)

            # lidar projections
            cam2 = cam2[0].cpu().numpy()
            velo = velo[0].cpu().numpy()
            T = T[0].cpu().numpy()
            P = P[0].cpu().numpy()
            gt_err = gt_err.numpy()[0]

            rot_err[i, 3:] = gt_err[:3]
            trans_err[i, 3:] = gt_err[3:]

            T_err = phi_to_transformation_matrix(gt_err)
            T_composed = T_err @ T
            uncalib = lidar_projection(velo, T_composed, P, cam2.shape[:2], downscale=downscale)

            T_est = phi_to_transformation_matrix(np.concatenate([pred_r[0], pred_t[0]]))
            T_patch = inv_transform(T_est)
            T_recalib = T_patch @ T_composed
            recalib = lidar_projection(velo, T_recalib, P, cam2.shape[:2], downscale=downscale)
            calibgt = lidar_projection(velo, T, P, cam2.shape[:2], downscale=downscale)

            # calculate running msee
            running_msee += np.linalg.norm((T_est - T_err))

            # show img
            if downscale != 1:
                cam2 = cv2.resize(cam2, (w, h), interpolation=cv2.INTER_AREA)
            uncalib = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), uncalib),
                                  'Input', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            recalib = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), recalib),
                                  'Pred', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            calibgt = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), calibgt),
                                  'GT', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            img = np.concatenate([uncalib, recalib, calibgt], axis=0)
            cv2.imshow('Visualization', img)
            if fout is not None:
            #     writer.write(img)
                cv2.imwrite(Path(fout).joinpath('img').joinpath(f'{i:04d}.png').as_posix(), img)
            key = cv2.waitKey(int(1 / 60 * 1000))
            if (key & 0xFF == ord('q')) or (key & 0xFF == 27):
                break

    # if fout is not None:
    #     writer.release()
    # show results
    avg_inference_time = running_time / (i + 1)
    print(f'Running loss: {running_mean:.04f}')
    print(f'MSEE: {running_msee/(i + 1):.04f}')
    print(f'Avg. inference speed: {avg_inference_time:.04f} s')
    print(f'Rotation mean errors (degree): {np.rad2deg(rot_err.mean(axis=0)[:3])}')
    print(f'Rotation max errors (degree): {np.rad2deg(rot_err.max(axis=0)[:3])}')
    print(f'Rotation min errors (degree): {np.rad2deg(rot_err.min(axis=0)[:3])}')
    print(f'Rotation std errors (degree): {np.rad2deg(rot_err.std(axis=0)[:3])}')
    print(f'Translation mean errors (meter): {trans_err.mean(axis=0)[:3]}')
    print(f'Translation max errors (meter): {trans_err.max(axis=0)[:3]}')
    print(f'Translation min errors (meter): {trans_err.min(axis=0)[:3]}')
    print(f'Translation std errors (meter): {trans_err.std(axis=0)[:3]}')

    # if fout is not None:
    #     pd.DataFrame(rot_err).to_csv(Path(fout).joinpath('rot_stat.csv'), header=['row', 'pitch', 'yaw', 'row_gt', 'pitch_gt', 'yaw_gt'], index=False)
    #     pd.DataFrame(trans_err).to_csv(Path(fout).joinpath('trans_stat.csv'), header=['x', 'y', 'z', 'x_gt', 'y_gt', 'z_gt'], index=False)
    return


def run_iterative(models: List[Module], loader: DataLoader, loss_fn: Loss, offsets: List[Tuple[float, float]], fout: Optional[str] = None) -> None:
    if fout is not None:
        pout = Path(fout).joinpath('model_inference.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(pout.as_posix(), fourcc, 10.0, (1242, 375*3))

    cv2.namedWindow('Visualization', flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Visualization', 1250, 380*3)

    running_loss = 0
    running_mean = 0
    running_time = 0
    running_msee = 0
    rot_err = np.zeros((len(loader.dataset), 3), np.float32)
    trans_err = np.zeros((len(loader.dataset), 3), np.float32)
    pbar = tqdm(loader)
    for m in models:
        m.eval()

    with torch.no_grad():
        i = 0
        for i, sample in enumerate(pbar):
            cam2_d, velo_d, gt_err, gt_err_norm, cam2, velo, T, P = sample[0].cuda(), sample[1].cuda(), sample[2], sample[3].cuda(), \
                                                              sample[4].cuda(), sample[5].cuda(), sample[6].cuda(), sample[7].cuda()
            # prepare matrix
            cam2 = cam2[0].cpu().numpy()
            velo_np = velo[0].cpu().numpy()
            T_np = T[0].cpu().numpy()
            P = P[0].cpu().numpy()
            gt_err = gt_err.numpy()[0]
            T_err = phi_to_transformation_matrix(gt_err)
            T_composed = T_err @ T_np

            # timer start
            t = time.time()

            # run iterative mode
            phi = T_composed.copy()
            T_est_final = np.eye(4, dtype=np.float32)
            for m, (rot_offset, trans_offset) in zip(models, offsets):
                pred_r, pred_t = m(velo_d, cam2_d)
                pred_r_real, pred_t_real = loader.dataset.destandardize(pred_r.detach().cpu().numpy(), pred_t.detach().cpu().numpy(),
                                                                        rot_offset=rot_offset, trans_offset=trans_offset)
                T_est = phi_to_transformation_matrix(np.concatenate([pred_r_real[0], pred_t_real[0]]))
                T_est_final = T_est @ T_est_final
                T_patch = inv_transform(T_est)
                phi = T_patch @ phi
                velo_d = torch.from_numpy(lidar_projection(velo_np, phi, P, cam2.shape[:2], crop=cam2_d.shape[2:])).unsqueeze(0).unsqueeze(0).cuda()
            T_recalib = phi.copy()

            # timer stop
            running_time += (time.time() - t)
            loss = loss_fn((pred_r, pred_t), torch.split(gt_err_norm, 3, dim=1), velo, T)

            # collect statistics
            running_loss += loss.item()
            running_mean = running_loss / (i + 1)
            pbar.set_description("Valid running loss: %.4f" % (running_mean))

            gt_splits = np.split(gt_err, 2, axis=0)
            rot_err[i] = np.abs(pred_r_real[0]-gt_splits[0]).mean(axis=0)
            trans_err[i] = np.abs(pred_t_real[0]-gt_splits[1]).mean(axis=0)

            # lidar projections -- raw errors
            uncalib = lidar_projection(velo_np, T_composed, P, cam2.shape[:2])
            # lidar projections -- model predictions
            recalib = lidar_projection(velo_np, T_recalib, P, cam2.shape[:2])
            # lidar projections -- ground truth
            calibgt = lidar_projection(velo_np, T_np, P, cam2.shape[:2])

            # calculate running msee
            running_msee += np.linalg.norm((T_est_final - T_err))

            # show img
            uncalib = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), uncalib),
                                  'Input', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            recalib = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), recalib),
                                  'Pred', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            calibgt = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), calibgt),
                                  'GT', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            img = np.concatenate([uncalib, recalib, calibgt], axis=0)
            cv2.imshow('Visualization', img)
            if fout is not None:
                writer.write(img)
            key = cv2.waitKey(int(1 / 60 * 1000))
            if (key & 0xFF == ord('q')) or (key & 0xFF == 27):
                break

    if fout is not None:
        writer.release()
    # show results
    avg_inference_time = running_time / (i + 1)
    print(f'Running loss: {running_mean:.04f}')
    print(f'MSEE: {running_msee/(i + 1):.04f}')
    print(f'Avg. inference speed: {avg_inference_time:.04f} s')
    print(f'Rotation mean errors (degree): {np.rad2deg(rot_err.mean(axis=0))}')
    print(f'Rotation max errors (degree): {np.rad2deg(rot_err.max(axis=0))}')
    print(f'Rotation min errors (degree): {np.rad2deg(rot_err.min(axis=0))}')
    print(f'Rotation std errors (degree): {np.rad2deg(rot_err.std(axis=0))}')
    print(f'Translation mean errors (meter): {trans_err.mean(axis=0)}')
    print(f'Translation max errors (meter): {trans_err.max(axis=0)}')
    print(f'Translation min errors (meter): {trans_err.min(axis=0)}')
    print(f'Translation std errors (meter): {trans_err.std(axis=0)}')
    return


def arg_parser():
    parser = argparse.ArgumentParser()
    # common args
    parser.add_argument('--dataset', type=str, default='/home/username/dataset/KITTI/', help='Model learning rate.')
    parser.add_argument('--model', type=int, default=1, help='Select model variant to test.')
    parser.add_argument('--loss_a', type=float, default=1.0, help='Loss factor for rotation & translation errors.')
    parser.add_argument('--loss_b', type=float, default=2.0, help='Loss factor for point cloud center errors.')
    parser.add_argument('--loss_c', type=float, default=2.0, help='Loss factor for point cloud errors.')

    # run visualization task
    parser.add_argument('--visualization', action='store_true', help='Show online running test.')
    parser.add_argument('--ckpt', type=str, help='Path to the saved model in saved folder.')
    parser.add_argument('--rotation_offsest', type=float, default=10.0, help='Random rotation error range.')
    parser.add_argument('--translation_offsest', type=float, default=0.2, help='Random translation error range.')

    # run in iterative mode
    parser.add_argument('--iterative', action='store_true', help='Show online iterative running test.')
    parser.add_argument('--ckpt_list', type=str, help='One or more paths to the saved model in saved folder. The first one will be used first.', nargs='+')
    parser.add_argument('--rotation_offsests', type=float, default=[10.0, 2.0], help='List of random rotation error range.', nargs='+')
    parser.add_argument('--translation_offsests', type=float, default=[0.2, 0.2], help='List of random translation error range.', nargs='+')

    # output path
    parser.add_argument('--out_path', type=str, help='Path to store the visualized video.')
    # python eval.py --ckpt ../results/HOPE2/ckpt/Epoch47_val_0.0475.tar --visualization --rotation_offsest 10 --translation_offsest 0.2
    # python eval.py --ckpt_list ../results/HOPE-10-0.2/ckpt/Epoch65_val_0.2983.tar ../results/HOPE2/ckpt/Epoch47_val_0.0475.tar --iterative --rotation_offsest 10 --translation_offsest 0.2
    return parser.parse_args()


if __name__ == '__main__':
    # parse args
    args = arg_parser()
    print(args)

    # dataset & loss
    test_ds = CalibDataset(path=args.dataset, mode='test', rotation_offset=args.rotation_offsest, translation_offset=args.translation_offsest)
    criteria = Loss(dataset=test_ds, alpha=args.loss_a, beta=args.loss_b, gamma=args.loss_c)

    # run
    if args.visualization:
        # build data loader
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4,
                                 pin_memory=True, drop_last=False)

        # build model
        model = get_model(args.model)
        print(f"Model trainable parameters: {count_parameters(model)}")

        # load checkpoint
        if args.ckpt:
            ckpt = torch.load(args.ckpt)
            model.load_state_dict(ckpt['model'])

            trained_epochs = ckpt['epoch']
            global_i = ckpt['global_i']
            best_val = ckpt['best_val']
            train_loss = ckpt['loss']
            try:
                rng = ckpt['rng']
                torch.set_rng_state(rng)
            except KeyError as e:
                print('No rng state in the checkpoint.')
            print(f'Model loaded. '
                  f'Trained epochs: {trained_epochs}; global i: {global_i}; '
                  f'train loss: {train_loss:.04f}; best validation loss: {best_val:.04f}.')
        else:
            print('A model checkpoint must be provided to run the test.')
            exit(0)

        # push model to gpu
        model.cuda()

        # run task
        run_visualization(model, test_loader, criteria, args.out_path)
    elif args.iterative:
        # build data loader
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4,
                                 pin_memory=True, drop_last=False)

        # build models
        assert len(args.ckpt_list) != 0, "List of checkpoint files must be provided to run in iterative mode."
        models = []
        offsets = []
        for i, fckpt in enumerate(args.ckpt_list):
            print(f'Loading model {i+1}...')
            model = get_model(args.model)
            ckpt = torch.load(fckpt)

            model.load_state_dict(ckpt['model'])
            model.cuda()
            models.append(model)
            offsets.append((args.rotation_offsests[i], args.translation_offsests[i]))

            trained_epochs = ckpt['epoch']
            global_i = ckpt['global_i']
            best_val = ckpt['best_val']
            train_loss = ckpt['loss']
            try:
                rng = ckpt['rng']
                torch.set_rng_state(rng)
            except KeyError as e:
                print('No rng state in the checkpoint.')
            print(f'Model loaded. '
                  f'Trained epochs: {trained_epochs}; global i: {global_i}; '
                  f'train loss: {train_loss:.04f}; best validation loss: {best_val:.04f};'
                  f'Offsets: {args.rotation_offsests[i]} degrees, {args.translation_offsests[i]} meters.')

        # run task
        run_iterative(models, test_loader, criteria, offsets, args.out_path)
    else:
        raise NotImplementedError
