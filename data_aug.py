import numpy as np


def jitter(x, sigma=0.03):
    """
    The function generates noise by sampling from a normal distribution with mean 0 and standard deviation sigma, and adds this noise to the input time series.
    It then returns the augmented time series.

    https://arxiv.org/pdf/1706.00527.pdf
    """
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def permutation(x, max_segments=5, seg_mode="equal"):
    """
    This function performs data augmentation on a time series by randomly permuting segments of the time series.
    The function first generates a random number of segments for each time series using np.random.randint.
    If seg_mode is 'equal', the time series is divided into an equal number of segments.
    If seg_mode is 'random', the time series is divided into random segments by selecting a random number of split points.
    The segments are then randomly permuted and concatenated to form the augmented time series.
    The function returns the augmented time series.
    """

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
    """
    This function is used for data augmentation of a time series dataset. It takes four arguments:
    x: a 2D array of shape (num_samples, sequence_length) representing the time series data.
    y: a 1D array of shape (num_samples,) representing the labels for each time series in x.
    aug_ratio: an integer representing the number of augmented samples to generate for each class with less than 50 samples.
    method: a string representing the method to use for data augmentation. It can be either 'jitter' or 'permutation'.
    The function generates augmented samples for each class that has less than 50 samples using the specified method and returns the augmented time series data and labels.
    """
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

    return x_aug, y_aug
