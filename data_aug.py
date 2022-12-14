import numpy as np
import os
from data_utils import plot_sample

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def timeseries_aug(x, y, aug_ratio, method):
    x_aug = x
    y_aug = y
    if aug_ratio > 0:
        for n in range(aug_ratio):
            if (method == 'jitter'):
                x_temp = jitter(x)
            if (method == 'scaling'):
                x_temp=scaling(x)
            if (method == 'rotation'):
                x_temp=rotation(x)
            if (method == 'permutation'):
                x_temp=permutation(x)

            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y, axis=0)

    plot_sample(x)
    plot_sample(x_aug)

    return x_aug, y_aug

