# reference: https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8
import numpy as np
from scipy.ndimage.filters import maximum_filter

def nms_2d(hough_space, wsize=40):
    """
    Algorithm:
        1. construct a window with a size of wsize x wsize
        2. start in the first pixel of the hough_space
        3. surround the pixel with the window and find the maximum value in the area
        4. substitute the value of the pixel with the maximum value
        5. slide the window one pixel
        6. repeat step 3-5 until we have covered the entire heatmap
        7. compare the origin hough_space
        8. output the pixels staying with the same value
    """
    nms_map = maximum_filter(hough_space, footprint=np.ones(shape=[wsize, wsize]))
    inds = np.argwhere(hough_space == nms_map)
    return inds[nms_map[inds[:, 0], inds[:, 1]] != 0]

