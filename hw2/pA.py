import argparse
from pathlib import Path

import numpy as np


class ProjectionMatrixSolver:
    def __init__(self, method='p34=1'):
        '''
        Args:
            method: (str) should be one of 'norm=1', 'sum=1', 'p34=1'
        '''
        self.P = None
        self.K = None
        self.R = None
        self.t = None
        assert method in ['norm=1', 'sum=1', 'p34=1']
        self.method = method

    def fit(self, pt2d, pt3d):
        '''
        Args:
            pt2d: (ndarray) sized [N, 2]
            pt3d: (ndarray) sized [N, 3]
        '''
        N = pt2d.shape[0]
        x, y = pt2d.T
        X, Y, Z = pt3d.T

        A = np.zeros((2 * N, 12))
        val1 = np.full((N,), 1.0)
        val0 = np.full((N,), 0.0)
        A[0::2] = np.stack(
            [X, Y, Z, val1, val0, val0, val0, val0, -x * X, -x * Y, -x * Z, -x], axis=1
        )
        A[1::2] = np.stack(
            [val0, val0, val0, val0, X, Y, Z, val1, -y * X, -y * Y, -y * Z, -y], axis=1
        )

        if self.method == 'norm=1':
            W, V = np.linalg.eig(A.T @ A)
            P = V[:, W.argmin()].reshape(3, 4)
        if self.method == 'sum=1':
            a = np.concatenate([A, np.ones((1, 12))], axis=0)
            b = np.zeros((A.shape[0] + 1))
            b[-1] = 1.0
            P = (np.linalg.pinv(a) @ b).reshape(3, 4)
        if self.method == 'p34=1':
            a = A[:, :-1]
            b = -A[:, -1]
            P = np.concatenate([(np.linalg.pinv(a) @ b), np.ones(1)]).reshape(3, 4)

        P = P * np.sign(np.linalg.det(P[:, :3]))
        M, P4 = P[:, :3], P[:, 3]
        q, r = np.linalg.qr(np.linalg.inv(M))
        K = np.linalg.inv(r)
        R = np.linalg.inv(q)

        D = np.diag(np.sign(np.diag(K)))
        K = K @ D
        R = D @ R
        t = np.linalg.inv(K) @ P4
        K = K / K[-1, -1]

        self.P = P
        self.K = K
        self.R = R
        self.t = t

    def predict(self, pt3d):
        P = self.K @ np.concatenate([self.R, self.t.reshape(3, 1)], axis=1)
        homo3d = np.concatenate([pt3d.T, np.ones((1, pt3d.shape[0]))], axis=0)
        homo2d = P @ homo3d
        coor2d = homo2d[:2, :] / homo2d[2, :]
        pt2d = coor2d.T
        return pt2d

    def check(self):
        eps = 1e-7
        assert np.allclose(self.K, np.triu(self.K), eps)  # K is upper triangular
        assert np.alltrue(np.diag(self.K) > 0)  # K diagonal all positive
        assert np.allclose(self.R.T @ self.R, np.eye(3), eps)  # R is orthogonal
        assert np.allclose(np.linalg.det(self.R), 1.0, eps)  # det(R) == 1

    def __str__(self):
        return 'P\n{}\nK\n{}\nR\n{}\nt\n{}'.format(
            np.round(self.P, 3),
            np.round(self.K, 3),
            np.round(self.R, 3),
            np.round(self.t, 3),
        )

    @staticmethod
    def rmse(pred2d, true2d):
        return np.sqrt(((pred2d - true2d) ** 2).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assets_dir', help='Directory of assets', type=Path)
    args = parser.parse_args()

    assert args.assets_dir.exists()
    img_path_1 = args.assets_dir / 'chessboard_1.jpg'
    img_path_2 = args.assets_dir / 'chessboard_2.jpg'
    pt2d_path_1 = args.assets_dir / 'box1.npy'
    pt2d_path_2 = args.assets_dir / 'box2.npy'
    pt3d_path = args.assets_dir / 'Point3D.txt'

    assert img_path_1.exists()
    assert img_path_2.exists()
    assert pt2d_path_1.exists()
    assert pt2d_path_2.exists()
    assert pt3d_path.exists()

    pt3d = np.loadtxt(pt3d_path)
    pt2d_1 = np.load(pt2d_path_1)
    pt2d_2 = np.load(pt2d_path_2)

    proj1 = ProjectionMatrixSolver(method='norm=1')
    proj1.fit(pt2d_1, pt3d)
    proj1.check()
    pred_1 = proj1.predict(pt3d)
    error = proj1.rmse(pred_1, pt2d_1)
    print(proj1)
    print('error:', error)
    print('-' * 50)

    proj2 = ProjectionMatrixSolver(method='norm=1')
    proj2.fit(pt2d_2, pt3d)
    proj2.check()
    pred_2 = proj2.predict(pt3d)
    error = proj2.rmse(pred_2, pt2d_2)
    print(proj2)
    print('error:', error)
    print('-' * 50)

    from visualize import visualize

    visualize(pt3d, proj1.R, proj1.t.reshape(3, 1), proj2.R, proj2.t.reshape(3, 1))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(plt.imread(img_path_1))
    ax[0].plot(pt2d_1[:, 0], pt2d_1[:, 1], 'r.')
    ax[0].plot(pred_1[:, 0], pred_1[:, 1], 'b.')
    ax[1].imshow(plt.imread(img_path_2))
    ax[1].plot(pt2d_2[:, 0], pt2d_2[:, 1], 'r.')
    ax[1].plot(pred_2[:, 0], pred_2[:, 1], 'b.')
    plt.show()
