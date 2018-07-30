# reference: http://ais.informatik.uni-freiburg.de/teaching/ss11/robotics/slides/17-icp.pdf
# reference: https://github.com/ClayFlannigan/icp/blob/master/icp.py
# reference: https://gist.github.com/ecward/c373932638fd04a2243e
# reference: https://github.com/opencv/opencv_contrib/blob/master/modules/surface_matching/include/opencv2/surface_matching/icp.hpp

import numpy as np
import matplotlib.pyplot as plt

NUM_SRC = 60
NUM_DST = 520


def rotation_matrix(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.cos(alpha)]])


def create_dataset(vis=False):
    """
    Algorithm:
      1. generate the data pool acoording to 'y = sin(x)'
      2. transform the data to get the destination data: y' = r * y + t
      3. sample the data to produce the source data
    """
    x = np.linspace(start=-np.pi / 2, stop=np.pi / 2, num=NUM_DST)
    y = np.sin(x)
    data = np.stack([x, y], axis=1)

    alpha = np.pi / 2
    r = rotation_matrix(alpha)
    t = np.array([1.2, 0.5])
    dst = np.dot(data, r.T) + t

    inds = np.random.choice(NUM_DST, size=NUM_SRC, replace=False)
    src = data[inds]

    if vis:
        plt.plot(src[:, 0], src[:, 1], '.')
        plt.plot(dst[:, 0], dst[:, 1], 'r.')
        plt.show()

    return src, dst


if __name__ == '__main__':
    src, dst = create_dataset()

