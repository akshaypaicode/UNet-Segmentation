import numpy as np
from keras.utils import to_categorical


def one_hot_encode_y(y, n_classes=135):
    shape = y.shape
    y = to_categorical(y, num_classes=n_classes).astype(np.uint8)
    y = y.reshape(shape + (n_classes,))
    return y
