import random
import numpy as np
import argparse
import torch
from torch import Tensor
import torch.utils.data
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from datetime import datetime
import shutil
from net.CalibrationNet import get_model
from dataset import TrainDataset as CalibDataset, run_stat
from utils import count_parameters, inv_transform_vectorized, phi_to_transformation_matrix_vectorized

# set seeds
SEED = 53
rng = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda'


class Loss(Module):
    def __init__(self, dataset: CalibDataset, reduction: str = 'mean',
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        super(Loss, self).__init__()
        self.dataset = dataset

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
        min_pts = []
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
            min_pts.append(pts3d.shape[0])
            # project
            pts3d_cam = T_.mm(pts3d.t()).t().unsqueeze(0)
            pts3d_cam_recalib = T_recalib_.mm(pts3d.t()).t().unsqueeze(0)
            pts3d_cam_l.append(pts3d_cam)
            pts3d_cam_recalib_l.append(pts3d_cam_recalib)

        min_pts = min(min_pts)
        pts3d_cam_l = [pts[:, :min_pts, :] for pts in pts3d_cam_l]
        pts3d_cam_recalib_l = [pts[:, :min_pts, :] for pts in pts3d_cam_recalib_l]
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


def train_one_epoch(model: Module, loss_fn: Loss, opt: Optimizer, loader: DataLoader,
                    epoch: int, global_i: int, logger: SummaryWriter):
    running_loss = 0
    running_mean = 0
    pbar = tqdm(loader)
    model.train()

    for i, sample in enumerate(pbar):
        cam2_d, velo_d, gt_err, gt_err_norm, velo, T = sample[0].cuda(), sample[1].cuda(), sample[2], \
                                                       sample[3].cuda(), sample[4].cuda(), sample[5].cuda()

        # forward
        pred_r, pred_t = model(velo_d, cam2_d)
        loss = loss_fn((pred_r, pred_t), torch.split(gt_err_norm, 3, dim=1), velo, T)

        # backward
        # zero the parameter gradients
        opt.zero_grad()
        # auto-calculate gradients
        loss.backward()
        # apply gradients
        opt.step()

        # collect statistics
        running_loss += loss.item()
        running_mean = running_loss / (i + 1)
        pbar.set_description("Epoch %d, train running loss: %.4f" % (epoch + 1, running_mean))

        # log statistics
        if (i+1) % 100 == 0:
            gt_splits = np.split(gt_err.numpy(), 2, axis=1)  # split into 2 parts
            pred_r, pred_t = loader.dataset.destandardize(pred_r.detach().cpu().numpy(), pred_t.detach().cpu().numpy())
            err_a, err_b, err_c = np.abs(pred_r-gt_splits[0]).mean(axis=0)
            err_x, err_y, err_z = np.abs(pred_t-gt_splits[1]).mean(axis=0)
            logger.add_scalar('Train/Loss/loss', running_mean, global_i)
            logger.add_scalar('Train/Loss/roll', err_a, global_i)
            logger.add_scalar('Train/Loss/pitch', err_b, global_i)
            logger.add_scalar('Train/Loss/yaw', err_c, global_i)
            logger.add_scalar('Train/Loss/x', err_x, global_i)
            logger.add_scalar('Train/Loss/y', err_y, global_i)
            logger.add_scalar('Train/Loss/z', err_z, global_i)
        global_i += 1
    return running_mean, global_i


def validation(model: Module, loss_fn: Loss, loader: DataLoader,
               epoch: int, global_i: int, logger: SummaryWriter):
    running_loss = 0
    running_mean = 0
    rot_err = np.zeros(3, np.float32)
    trans_err = np.zeros(3, np.float32)
    pbar = tqdm(loader)
    model.eval()

    with torch.no_grad():
        i = 0
        for i, sample in enumerate(pbar):
            cam2_d, velo_d, gt_err, gt_err_norm, velo, T = sample[0].cuda(), sample[1].cuda(), sample[2], \
                                                           sample[3].cuda(), sample[4].cuda(), sample[5].cuda()
            pred_r, pred_t = model(velo_d, cam2_d)
            loss = loss_fn((pred_r, pred_t), torch.split(gt_err_norm, 3, dim=1), velo, T)

            # collect statistics
            running_loss += loss.item()
            running_mean = running_loss / (i + 1)
            pbar.set_description("Epoch %d, valid running loss: %.4f" % (epoch + 1, running_mean))

            gt_splits = np.split(gt_err.numpy(), 2, axis=1)
            pred_r, pred_t = loader.dataset.destandardize(pred_r.detach().cpu().numpy(), pred_t.detach().cpu().numpy())
            rot_err += np.abs(pred_r-gt_splits[0]).mean(axis=0)
            trans_err += np.abs(pred_t-gt_splits[1]).mean(axis=0)

        # write results
        rot_err = rot_err / (i + 1)
        trans_err = trans_err / (i + 1)
        logger.add_scalar('Valid/Loss/loss', running_mean, global_i)
        logger.add_scalar('Valid/Loss/roll', rot_err[0], global_i)
        logger.add_scalar('Valid/Loss/pitch', rot_err[1], global_i)
        logger.add_scalar('Valid/Loss/yaw', rot_err[2], global_i)
        logger.add_scalar('Valid/Loss/x', trans_err[0], global_i)
        logger.add_scalar('Valid/Loss/y', trans_err[1], global_i)
        logger.add_scalar('Valid/Loss/z', trans_err[2], global_i)
    return running_mean


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/username/dataset/KITTI/', help='Model learning rate.')
    parser.add_argument('--batch', type=int, default=2, help='Batch size.')
    parser.add_argument('--ckpt', type=str, help='Path to the saved model in saved folder.')
    parser.add_argument('--ckpt_no_lr', action='store_true', help='Ignore lr in the checkpoint.')
    parser.add_argument('--model', type=int, default=1, help='Select model variant to test.')
    parser.add_argument('--rotation_offsest', type=float, default=10.0, help='Random rotation error range.')
    parser.add_argument('--translation_offsest', type=float, default=0.2, help='Random translation error range.')
    parser.add_argument('--epoch', type=int, default=50, help='Epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Model learning rate.')
    parser.add_argument('--patience', type=int, default=6, help='Patience for reducing lr.')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor for reducing lr.')
    parser.add_argument('--loss_a', type=float, default=1.0, help='Loss factor for rotation & translation errors.')
    parser.add_argument('--loss_b', type=float, default=1.0, help='Loss factor for point cloud center errors.')
    parser.add_argument('--loss_c', type=float, default=1.0, help='Loss factor for point cloud errors.')
    parser.add_argument('--exp_name', type=str, default=f'exp_{datetime.now().strftime("%H%M_%d%m%Y")}',
                        help='Loss factor for translation errors.')
    parser.add_argument('--stat', action='store_true', help='Calculate dataset statistics.')
    return parser.parse_args()


if __name__ == '__main__':
    # parse args
    args = arg_parser()
    print(args)

    # calculating dataset statistics
    if args.stat:
        train_ds = CalibDataset(path=args.dataset, mode='train')
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        run_stat(train_loader)
        exit(0)

    # creating dirs
    result_base = Path('.').absolute().parent.joinpath('results').joinpath(args.exp_name)
    log_dir = result_base.joinpath('log')
    ckpt_dir = result_base.joinpath('ckpt')
    mod_dir = result_base.joinpath('mod')
    result_base.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    mod_dir.mkdir(exist_ok=True)
    # save critical files
    shutil.copy2('train.py', mod_dir.as_posix())
    shutil.copy2('train.sh', mod_dir.as_posix())
    shutil.copy2('./net/CalibrationNet.py', mod_dir.as_posix())
    shutil.copy2('./net/Convolution.py', mod_dir.as_posix())
    shutil.copy2('./net/SpatialPyramidPooling.py', mod_dir.as_posix())

    # build model
    model = get_model(args.model)
    print(f"Model trainable parameters: {count_parameters(model)}")

    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.patience,
                                  verbose=True, min_lr=1e-8, cooldown=2)

    # move model to gpu
    model.cuda()

    # load ckpt
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['scheduler'])

        trained_epochs = ckpt['epoch']
        global_i = ckpt['global_i']
        best_val = ckpt['best_val']
        train_loss = ckpt['loss']
        rng = ckpt['rng']
        torch.set_rng_state(rng)

        if args.ckpt_no_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    else:
        trained_epochs = 0
        global_i = 0
        best_val = 9999
        train_loss = 9999
    valid_loss = 9999

    # create data loaders
    train_ds = CalibDataset(path=args.dataset, mode='train',
                            rotation_offset=args.rotation_offsest, translation_offset=args.translation_offsest)
    val_ds = CalibDataset(path=args.dataset, mode='val',
                          rotation_offset=args.rotation_offsest, translation_offset=args.translation_offsest)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)

    # loss
    criteria = Loss(dataset=train_ds, alpha=args.loss_a, beta=args.loss_b, gamma=args.loss_c)

    # summary writer
    writer = SummaryWriter(log_dir=log_dir.as_posix(), purge_step=global_i)

    # train model
    for epoch in range(trained_epochs, trained_epochs+args.epoch):
        # train
        train_loss, global_i = train_one_epoch(model, criteria, optimizer, train_loader, epoch, global_i, writer)

        # valid
        valid_loss = validation(model, criteria, val_loader, epoch, global_i, writer)

        if valid_loss < best_val:
            print(f'Best model (val:{valid_loss:.04f}) saved.')
            best_val = valid_loss
            torch.save({
                'epoch': epoch+1,
                'global_i': global_i,
                'best_val': best_val,
                'loss': train_loss,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'rng': torch.get_rng_state()
            }, ckpt_dir.joinpath(f'Epoch{epoch+1}_val_{best_val:.04f}.tar').as_posix())

        # update scheduler
        scheduler.step(valid_loss)

    # final model
    torch.save({
        'epoch': trained_epochs+args.epoch,
        'global_i': global_i,
        'best_val': best_val,
        'loss': train_loss,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng': torch.get_rng_state()
    }, ckpt_dir.joinpath(f'Epoch{trained_epochs+args.epoch}_val_{valid_loss:.04f}.tar').as_posix())
