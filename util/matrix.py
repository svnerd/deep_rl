
import numpy as np

def to_2d(array):
    if array.ndim >= 3:
        raise Exception(">2d matrix. cannot handle")
    if array.ndim == 1:
        return np.reshape(array, (1, -1))
    return array