import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import io
from skimage import util
from skimage import color
from skimage import filters
from skimage import transform


class HarrisCorner:
    def __init__(
        self,
        gaussian_k=10,
        response_k=0.05,
        patch_window=15,
        nms_window=50,
        response_thresh=1e-8,
    ):
        self.gaussian_k = gaussian_k
        self.patch_window = patch_window
        self.response_k = response_k
        self.response_thresh = response_thresh
        self.nms_window = nms_window

    def apply(self, img):
        img = self.gaussian(img, k=self.gaussian_k)
        dy, dx = self.sobel(img)
        Axx, Axy, Ayy = self.structure_tensor(dy, dx, window=self.patch_window)
        R = self.harris_response(Axx, Axy, Ayy, k=self.response_k)
        rr, cc = self.peak_local_max(R, thresh=self.response_thresh, window=self.nms_window)
        return rr, cc

    @staticmethod
    def visualize_sobel(img):
        img = HarrisCorner.gaussian(img)
        dy, dx = HarrisCorner.sobel(img)
        mag = np.sqrt(dy ** 2 + dx ** 2)
        hsv = np.zeros((*img.shape, 3))
        hsv[..., 0] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
        hsv[..., 1] = 1.0
        hsv[..., 2] = (mag - mag.min()) / mag.max()
        return mag, color.hsv2rgb(hsv)

    @staticmethod
    def gaussian(img, k=10):
        sx, sy = 5, 5  # sigma
        mx, my = (0 + k - 1) / 2, (0 + k - 1) / 2  # mu
        xx = np.arange(k) - mx
        yy = np.arange(k) - my
        gx = np.exp(-0.5 * (xx / sx) ** 2) / (np.sqrt(2 * np.pi) * sx)
        gy = np.exp(-0.5 * (yy / sy) ** 2) / (np.sqrt(2 * np.pi) * sy)
        kernel = np.outer(gy, gx)
        kernel = kernel / kernel.sum()
        return ndi.convolve(img, kernel)

    @staticmethod
    def sobel(img):
        kernel = np.float64([[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]]) / 8.0
        dy = ndi.convolve(img, kernel)
        dx = ndi.convolve(img, kernel.T)
        return dy, dx

    @staticmethod
    def structure_tensor(dy, dx, window=10):
        Axx = HarrisCorner.gaussian(dx * dx, k=window)
        Axy = HarrisCorner.gaussian(dx * dy, k=window)
        Ayy = HarrisCorner.gaussian(dy * dy, k=window)
        return Axx, Axy, Ayy

    @staticmethod
    def harris_response(Axx, Axy, Ayy, k=0.05):
        det = Axx * Axy - Axy * Axy
        tr = Axx + Axy
        R = det - k * tr * tr
        return R

    @staticmethod
    def peak_local_max(R, thresh=1e-7, window=15):
        maxR = ndi.maximum_filter(R, size=window)
        mask1 = R > thresh
        mask2 = np.abs(maxR - R) < 1e-15
        return np.nonzero(mask1 & mask2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to image', type=Path)
    args = parser.parse_args()

    assert args.image_path.exists()

    img = io.imread(args.image_path)
    img = util.img_as_float(img)
    gray = color.rgb2gray(img)

    mag, vec = HarrisCorner.visualize_sobel(gray)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(mag)
    ax[2].imshow(vec)
    plt.show()

    test1 = gray
    test2 = transform.rotate(gray, 30.0)
    test3 = transform.rescale(gray, 0.5, multichannel=False)
    rr1, cc1 = HarrisCorner().apply(test1)
    rr2, cc2 = HarrisCorner().apply(test2)
    rr3, cc3 = HarrisCorner().apply(test3)

    fig, ax = plt.subplots(1, 3, dpi=180)
    ax[0].imshow(test1, cmap='gray')
    ax[1].imshow(test2, cmap='gray')
    ax[2].imshow(test3, cmap='gray')
    ax[0].plot(cc1, rr1, 'r+', markersize=3)
    ax[1].plot(cc2, rr2, 'r+', markersize=3)
    ax[2].plot(cc3, rr3, 'r+', markersize=3)
    ax[0].set_title('#Corner: {}'.format(rr1.shape[0]))
    ax[1].set_title('#Corner: {}'.format(rr2.shape[0]))
    ax[2].set_title('#Corner: {}'.format(rr3.shape[0]))
    fig.tight_layout()
    plt.show()
