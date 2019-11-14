import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage import io
from skimage import util
from skimage import draw

def forward_warp(H_src2dst, src_img, src_mask, dst_shape, fill=1.0):
    dstH, dstW = dst_shape

    src_yy, src_xx = np.nonzero(src_mask)
    src_homo = np.stack([src_xx, src_yy, np.ones_like(src_xx)], axis=0)
    dst_homo = H_src2dst @ src_homo.astype(np.float32)
    dst_xx, dst_yy = dst_homo[:2] / (dst_homo[2] + 1e-8)
    dst_xx = np.clip(np.round(dst_xx).astype(np.int32), 0, dstW - 1)
    dst_yy = np.clip(np.round(dst_yy).astype(np.int32), 0, dstH - 1)

    dst_img = np.full((dstH, dstW, 3), fill)
    dst_img[dst_yy, dst_xx] = src_img[src_yy, src_xx]
    return dst_img

def backward_warp(H_src2dst, src_img, dst_mask, dst_shape, fill=1.0):
    dstH, dstW = dst_shape
    srcH, srcW, _ = src_img.shape

    dst_yy, dst_xx = np.nonzero(dst_mask)
    dst_homo = np.stack([dst_xx, dst_yy, np.ones_like(dst_xx)], axis=0)
    src_homo = np.linalg.inv(H_src2dst) @ dst_homo.astype(np.float32)
    src_xx, src_yy = src_homo[:2] / (src_homo[2] + 1e-8)
    src_xx = np.clip(np.round(src_xx).astype(np.int32), 0, srcW - 1)
    src_yy = np.clip(np.round(src_yy).astype(np.int32), 0, srcH - 1)

    dst_img = np.full((dstH, dstW, 3), fill)
    dst_img[dst_yy, dst_xx] = src_img[src_yy, src_xx]
    return dst_img


class ProjectiveTransform:
    def __init__(self):
        pass

    def estimate(self, src, dst):
        '''
        Args:
            src: (ndarray) sized [N, 2]
            dst: (ndarray) sized [N, 2]
        '''
        N = src.shape[0]
        x_src, y_src = src.T
        x_dst, y_dst = dst.T

        A = np.zeros((2 * N, 9))
        val1 = np.full((N,), 1.0)
        val0 = np.full((N,), 0.0)

        A[0::2] = np.stack(
            [x_src, y_src, val1, val0, val0, val0, -x_dst * x_src, -x_dst * y_src, x_dst],
            axis=1,
        )
        A[1::2] = np.stack(
            [val0, val0, val0, x_src, y_src, val1, -y_dst * x_src, -y_dst * y_src, y_dst],
            axis=1,
        )
        a, b = A[:, :-1], A[:, -1]
        self.H = np.concatenate([np.linalg.pinv(a) @ b, np.ones(1)]).reshape(3, 3)


if __name__ == '__main__':
    pts = np.load('./assets/homo1.npy')
    src_pts = pts[:4]
    dst_pts = pts[4:]
    img = io.imread('./assets/homo1.jpg')
    img = util.img_as_float(img)
    imgH, imgW, _ = img.shape

    src_mask = np.zeros((imgH, imgW))
    src_mask[draw.polygon(src_pts[:, 1], src_pts[:, 0], shape=(imgH, imgW))] = 1.0
    dst_mask = np.zeros((imgH, imgW))
    dst_mask[draw.polygon(dst_pts[:, 1], dst_pts[:, 0], shape=(imgH, imgW))] = 1.0

    transform = ProjectiveTransform()
    transform.estimate(src_pts, dst_pts)
    H = transform.H
    invH = np.linalg.inv(H)

    np.set_printoptions(suppress=True)
    print('H')
    print(np.round(H, 3))
    print('inv(H)')
    print(np.round(invH, 3))

    forward_src2dst = forward_warp(H, img, src_mask, [imgH, imgW])
    forward_dst2src = forward_warp(invH, img, dst_mask, [imgH, imgW])

    backward_src2dst = backward_warp(H, img, dst_mask, [imgH, imgW])
    backward_dst2src = backward_warp(invH, img, src_mask, [imgH, imgW])

    src_mask = np.expand_dims(src_mask, axis=-1)
    dst_mask = np.expand_dims(dst_mask, axis=-1)
    bg_mask = (1 - src_mask) * (1 - dst_mask)

    result1 = img * bg_mask + forward_src2dst * dst_mask + forward_dst2src * src_mask
    result2 = img * bg_mask + backward_src2dst * dst_mask + backward_dst2src * src_mask

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(img)
    ax[0, 1].add_patch(Polygon(src_pts, color='red', fill=False))
    ax[0, 1].add_patch(Polygon(dst_pts, color='blue', fill=False))
    ax[1, 0].imshow(result1)
    ax[1, 1].imshow(result2)
    plt.show()

