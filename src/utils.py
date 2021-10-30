import torch
from torch import Tensor
import numpy as np
import logging
from typing import Optional, Union, Tuple


# get logger
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Utils')


def count_parameters(model):
    """Calculate number of total parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inv_transform(T):
    """
    The inverse matrix of a transformation matrix (from pose 2 -> pose 1)
    :param T: transformation (calibration) matrix, | R  T |   4*4 matrix
                                                   | 0  1 |
    :return: inversed transformation matrix, shape (4,4)
    """
    # rotation
    rot = T[:3, :3]
    # translation
    trans = T[:3, 3]
    # assemble inverse transformation matrix
    rt = rot.transpose()
    tt = -rt @ trans.reshape(1, -1).transpose()  # -R^{-1}*t

    res = np.zeros((4, 4), dtype=np.float32)
    res[:3, :3] = rt
    res[:3, 3] = tt[:, 0]
    res[3, 3] = 1
    return res


def inv_transform_vectorized(T: Tensor, device: str = 'cuda'):
    """
    The inverse matrix of a transformation matrix (from pose 2 -> pose 1), vectorized version
    :param T: transformation (calibration) matrix, | R  T |   N*4*4 matrix
                                                   | 0  1 |
    :return: inversed transformation matrix, shape (N, 4, 4)
    """
    bs = T.shape[0]

    # rotation
    rot = T[:, :3, :3].transpose(1, 2)  # (N, 3, 3)
    # translation
    trans = T[:, :3, 3]

    # re-assemble inverse tranformation
    Tinv = torch.cat([rot, -rot.bmm(trans.unsqueeze(2))], dim=2)  # -R^{-1}*t
    Tinv = torch.cat([Tinv, torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).repeat(bs, 1).view(bs, 1, 4)], dim=1)
    return Tinv


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


def angle_to_rotation_matrix_vectorized(rot):
    """
    Transform vector of Euler angles to rotation matrix (vectorized version)
    ref: http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html

    :param rot: euler angle PyTorch Tensor (shape (N, 3), each column represents roll-pitch-yaw or x-y-z respectively)
    :return: N*3*3 rotation matrix
    """
    # get batch of roll, pitch, yaw
    u, v, w = rot[:, 0], rot[:, 1], rot[:, 2]

    # calculate the intermediate values
    s_u, c_u = u.sin(), u.cos()
    s_v, c_v = v.sin(), v.cos()
    s_w, c_w = w.sin(), w.cos()
    a00 = c_v*c_w
    a01 = s_u*s_v*c_w-c_u*s_w
    a02 = s_u*s_w+c_u*s_v*c_w
    a10 = c_v*s_w
    a11 = c_u*c_w+s_u*s_v*s_w
    a12 = c_u*s_v*s_w-s_u*c_w
    a20 = -s_v
    a21 = s_u*c_v
    a22 = c_u*c_v

    row1 = torch.cat([a00.unsqueeze(1), a01.unsqueeze(1), a02.unsqueeze(1)], 1)
    row2 = torch.cat([a10.unsqueeze(1), a11.unsqueeze(1), a12.unsqueeze(1)], 1)
    row3 = torch.cat([a20.unsqueeze(1), a21.unsqueeze(1), a22.unsqueeze(1)], 1)

    return torch.cat([row1.unsqueeze(1), row2.unsqueeze(1), row3.unsqueeze(1)], 1)


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


def phi_to_transformation_matrix_vectorized(rot: Tensor, trans: Tensor, device: str = 'cuda'):
    """
    Transform calibration vector to calibration matrix (Numpy version)
    \theta_{calib} = [r_x,r_y,r_z,t_x,t_y,t_z]^T -> \phi_{calib} 4*4 matrix

    :param phi: calibration PyTorch Tensor (length 3 shape (3,), roll-pitch-yaw or x-y-z), output from calibration network

    :return: transformation matrix from Lidar coordinates to camera's frame
    """
    bs = rot.shape[0]
    # get rotation matrix
    rot_mat = angle_to_rotation_matrix_vectorized(rot)

    # create transformation matrix
    T = torch.cat([rot_mat, trans.view(bs, 3, 1)], dim=2)
    T = torch.cat([T, torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).repeat(bs, 1).view(bs, 1, 4)], dim=1)
    return T


def read_depth(img, sparse_val=0):
    """
    Convert a PIL image (mode="I") to the depth map (np.ndarray)

    :param img: PIL image object
    :param sparse_val: value to encode sparsity with
    :return: depth map in np.ndarray
    """
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    depth_png = np.array(img, dtype=np.float32)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)
    depth = depth_png / 256
    if sparse_val > 0:
        depth[depth_png == 0] = sparse_val
    return depth  # shape (H, W)


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
                     R: Optional[np.ndarray] = None, crop: Optional[Union[np.ndarray, Tuple[int, int]]] = None, downscale: int = 1) -> np.ndarray:
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
    pts2d = pts2d.transpose().round().astype(np.int32)//downscale
    xmin, ymin = 0, 0
    xmax, ymax = shape[1]//downscale, shape[0]//downscale
    mask = (xmin < pts2d[:, 0]) & (pts2d[:, 0] < xmax) & \
           (ymin < pts2d[:, 1]) & (pts2d[:, 1] < ymax)
    pts2d = pts2d[mask][:, :2]  # keep only coordinates
    pts3d = pts3d[mask][:, 0]  # keep only x values of scan (depth)

    # draw depth map
    depth = np.zeros((shape[0]//downscale, shape[1]//downscale), dtype=np.float32)
    depth[pts2d[:, 1], pts2d[:, 0]] = pts3d

    # crop
    if crop is not None:
        depth = crop_img(depth, crop[0], crop[1])
    return depth


def map_to_range(val: np.ndarray, old_min: float, old_max: float, new_min: float, new_max: float):
    """
    map values from a range to another
    :param val: single int or float value or a list of values
    :param old_min: old values' minimum
    :param old_max: old values' maximum
    :param new_min: new values' minimum
    :param new_max: new values' maximum
    :return: mapped (float) values
    """
    old_range = old_max - old_min
    new_range = new_max - new_min
    valScaled = (val - old_min) / old_range  # to 0-1 range
    return new_min + (valScaled * new_range)  # to new range


def num_to_rgb(val: np.ndarray, max_val: float = 83, bgr: bool = True, ignore_zeros: bool = False):
    """
    map a list of values to RGB heat map
    :param val: list of values, List[int or float]
    :param max_val: float, maximum of given values
    :param bgr: if True, return BGR values instead of RGB
    :param ignore_zeros: if True, ignore mapping zeros
    :return: mapped color values, shape (N, 3), uint8
    """
    if np.any(val > max_val):
        max_val = np.max(val)
        # LOG.warning("[num_to_rgb] val %.2f is greater than max_val %.2f." % (np.max(val), max_val))
    if np.any(np.logical_or(val < 0, max_val < 0)):
        raise ValueError("arguments may not be negative")

    # available colors (RGB)
    colors = np.zeros((1024, 3), dtype=np.uint8)
    colors[:256, 0] = 255
    colors[:256, 1] = np.arange(256)
    colors[256:512, 0] = np.arange(255, -1, -1)
    colors[256:512, 1] = 255
    colors[512:768, 1] = 255
    colors[512:768, 2] = np.arange(256)
    colors[768:1024, 1] = np.arange(255, -1, -1)
    colors[768:1024, 2] = 255

    valScaled = np.floor(map_to_range(val, 0, max_val, 0, 1023)).astype(np.int32)

    mask = val == 0 if ignore_zeros else np.zeros(valScaled.shape[0], dtype=np.bool)

    new_val = colors[valScaled]
    new_val[mask] = 0

    if not bgr:
        # rgb
        return new_val
    else:
        # bgr
        new_val[:, [0, 1, 2]] = new_val[:, [2, 1, 0]]  # swap channels
        return new_val


def merge_color_img_with_depth(color: np.ndarray, depth: np.ndarray) -> np.ndarray:
    # crop color image to match the depth shape
    depth_shape = depth.shape
    color = crop_img(color, depth_shape[0], depth_shape[1])

    # calc depth mask
    mask = depth!=0

    rgbd = num_to_rgb(depth.flatten(), max_val=83, bgr=True, ignore_zeros=True).reshape(depth_shape+(3,))
    color[mask] = rgbd[mask]
    return color


# dump
def isRotationMatrix(R):
    """R*R^{-1}=I"""
    return torch.norm(torch.eye(3, dtype=R.dtype)-R.mm(R.t())) < 1e-6


def isRotationMatrix_np(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.matmul(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def cuda_memory_usage():
    LOG.warning('Cuda memory usage: %d' % torch.cuda.memory_allocated())
    return


def calc_crop_bbox(orig_shape, crop_shape):
    """
    Calculate the bounding box of the cropping image

    @param orig_shape: Tuple[int], a tuple of original shape values, (H1, W1)
    @param crop_shape: Tuple[int], a tuple of cropped shape values, (H2, W2), H2<H1, W2<W1
    @return: the bounding box boundaries, Hmin, Hmax, Wmin, Wmax
    """
    H, W = orig_shape
    h, w = crop_shape
    assert h < H and w < W, \
        'Crop size must be smaller than the original size! \n' \
        'Original shape ({o[0]},{o[1]}), Crop shape ({c[0]},{c[1]})'.format(o=orig_shape, c=crop_shape)
    hmin = H // 2 - (h // 2)
    wmin = W // 2 - (w // 2)
    return hmin, hmin+h, wmin, wmin+w


def center_crop(arr, crop_size):
    """
    Center crop an Numpy image

    @param arr: ndarray, image with shape (H,W) and (C,H,W), channel first.
    @param crop_size: Tuple[int], the target size/shape of the output
    @return: cropped image with crop_size
    """
    if len(arr.shape) == 2:
        H, W = arr.shape  # h -> y-axis, w -> x-axis
        dim = 2
    elif len(arr.shape) == 3:
        C, H, W = arr.shape  # h -> y-axis, w -> x-axis
        dim = 3
    elif len(arr.shape) == 4:
        B, C, H, W = arr.shape  # h -> y-axis, w -> x-axis
        dim = 4
    else:
        raise ValueError('The input array shape is not supported. Supported shape (H,W), (C,H,W), (B,C,H,W).')

    hmin, hmax, wmin, wmax = calc_crop_bbox((H, W), crop_size)

    if dim == 2:
        return arr[hmin:hmax, wmin:wmax]
    elif dim == 3:
        return arr[:, hmin:hmax, wmin:wmax]
    else:
        return arr[:, :, hmin:hmax, wmin:wmax]


def calib_np_to_phi(M, no_exception=False):
    """
    Transform calibration matrix to calibration vector (Numpy version)
    M_{calib} = | R  T |   4*4 matrix
                | 0  1 |

    :param M: 4*4 calibration matrix
    :param no_exception: bool, do not raise error there contains a wrong rotation matrix

    :return: calibration vector [r_x,r_y,r_z,t_x,t_y,t_z]
    """
    if M.shape[0] != 4 or M.shape[1] != 4:
        raise ValueError("A calibration (transformation) matrix must be a matrix of shape (4, 4)!")

    translation = np.transpose(M[:3, 3])
    orientation = rotation_matrix_to_angle(M[:3, :3], no_exception=no_exception)
    calib_vec = np.zeros(6, np.float32)
    calib_vec[:3] = orientation
    calib_vec[3:] = translation
    return calib_vec


def rotation_matrix_to_angle(R, no_exception=False):
    """
    Convert a 3*3 rotation matrix to roll, pitch, yaw
    :param R: 3*3 rotation matrix
    :param no_exception: bool, do not raise error when a wrong matrix comes
    :return: roll, pitch, yaw in radians
    """
    if not isRotationMatrix_np(R):
        if not no_exception:
            raise ValueError('This is not a rotation matrix! Hence can not convert to row, pitch, yaw!')
        else:
            # LOG.warning('This is not a rotation matrix! Hence can not convert to row, pitch, yaw!')
            return 0, 0, 0

    pitch = -np.arcsin(R[2, 0])

    if R[2,0] == 1:
        yaw = 0
        roll = np.arctan2(-R[0,1], -R[0,2])
    elif R[2,0] == -1:
        yaw = 0
        roll = np.arctan2(R[0,1], R[0,2])
    else:
        yaw = np.arctan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2,1], R[2,2])

    return roll, pitch, yaw
