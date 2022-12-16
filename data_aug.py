import numpy as np
import os
from data_utils import plot_sample


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


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
    train, counts_train = np.unique(y, return_counts=True)
    print("Before Augmentation:")
    for i in range(12):
        print('class: ', i, '\ttrain samples: ', counts_train[i])
        if (counts_train[i] < 51) and (aug_ratio > 0):
            for k in range(len(y)):
                if y[k] == i:
                    for n in range(aug_ratio):
                        if method == 'jitter':
                            x_temp = jitter(x[k])
                        if method == 'permutation':
                            x_temp = permutation(x[k])
                        x_temp = np.expand_dims(x_temp, axis=0)
                        y_temp = np.expand_dims(y[k], axis=0)
                        x_aug = np.append(x_aug, x_temp, axis=0)
                        y_aug = np.append(y_aug, y_temp, axis=0)

    print('\nAfter Augmentation:')
    train, counts_train_aug = np.unique(y_aug, return_counts=True)
    for i in range(12):
        print('class: ', i, '\ttrain samples: ', counts_train_aug[i])
    plot_sample(x)
    # plot_sample(x_aug)

    return x_aug, y_aug
