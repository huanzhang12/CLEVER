#!/usr/bin/env python3

import numpy as np
import time
from shmemarray import ShmemRawArray, NpShmemArray
from scipy.special import gammainc


"""
Original Matlab code:

function X = randsphere(m,n,r)

X = randn(m,n);
s2 = sum(X.^2,2);
X = X.*repmat(r*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);
"""

def randsphere_l2(idx, n, r, total_size, scale_size, tag_prefix, input_shape, X = None):
    result_arr = NpShmemArray(np.float32, (total_size, n), tag_prefix + "randsphere", False)
    # for scale, we may want a different starting point for imagenet, which is scale_start
    scale = NpShmemArray(np.float32, (scale_size, 1), tag_prefix + "scale", False)
    input_example = NpShmemArray(np.float32, input_shape, tag_prefix + "input_example", False)
    all_inputs = NpShmemArray(np.float32, (total_size,) + input_example.shape, tag_prefix + "all_inputs", False)
    # m is the number of items, off is the offset
    m, offset, scale_start = idx
    if X is None:
        # if X is not specified, then generate X as random gaussian
        # n is the dimension
        X = np.random.randn(m, n)
    else:
        # for testing
        # if X is specified, then set m and n accordingly
        m = X.shape[0]
        n = X.shape[1]
    s2 = np.sum(X * X, axis = 1)
    result_arr[offset : offset + m] = X * (np.tile(r*np.power(gammainc(n/2,s2/2), 1/n) / np.sqrt(s2), (n,1))).T
    # make a scaling
    result_arr[offset : offset + m] *= scale[offset + scale_start : offset + scale_start + m]
    # add to input example
    all_inputs[offset : offset + m] = input_example
    result_arr = result_arr.reshape(-1, *input_shape)
    all_inputs[offset : offset + m] += result_arr[offset : offset + m]
    return

def randsphere_li(idx, n, r, total_size, scale_size, tag_prefix, input_shape, X = None):
    result_arr = NpShmemArray(np.float32, (total_size, n), tag_prefix + "randsphere", False)
    # for scale, we may want a different starting point for imagenet, which is scale_start
    scale = NpShmemArray(np.float32, (scale_size, 1), tag_prefix + "scale", False)
    input_example = NpShmemArray(np.float32, input_shape, tag_prefix + "input_example", False)
    all_inputs = NpShmemArray(np.float32, (total_size,) + input_example.shape, tag_prefix + "all_inputs", False)
    # m is the number of items, off is the offset
    m, offset, scale_start = idx
    # generate random number uniformly from [-1.0, 1.0]
    # n is the dimension
    result_arr[offset : offset + m] = np.random.uniform(-1.0, 1.0, (m,n))
    # make a scaling
    result_arr[offset : offset + m] *= scale[offset + scale_start : offset + scale_start + m]
    # add to input example
    all_inputs[offset : offset + m] = input_example
    result_arr = result_arr.reshape(-1, *input_shape)
    all_inputs[offset : offset + m] += result_arr[offset : offset + m]
    return


if __name__ == "__main__":
    X = np.array([[0.20432, 0.74511], [0.87845, 0.52853], [0.96227, 0.69554]])
    import scipy.io as sio
    X = sio.loadmat('opt_matfiles/test.mat')['X']
    print(randsphere(3,2,0.5,X))

