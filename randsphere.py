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

def randsphere(idx, n, r, total_size, scale_size, tag_prefix, input_shape, X = None):
    """
    shared_data = ShmemRawArray('f', total_size * n, arr_tag, False)
    result_arr = np.ctypeslib.as_array(shared_data)
    result_arr = result_arr.reshape(total_size, n)
    """
    result_arr = NpShmemArray(np.float32, (total_size, n), tag_prefix + "randsphere", False)
    # for scale, we may want a different starting point for imagenet, which is scale_start
    scale = NpShmemArray(np.float32, (scale_size, 1), tag_prefix + "scale", False)
    input_example = NpShmemArray(np.float32, input_shape, tag_prefix + "input_example", False)
    all_inputs = NpShmemArray(np.float32, (total_size,) + input_example.shape, tag_prefix + "all_inputs", False)
    # m is the number of items, off is the offset
    m, offset, scale_start = idx
    if X is None:
        # if X is not specified, then generate X as random gaussian
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

"""
Batched version (abondaned)

import resource

def randsphere(m, n, r, X = None):
    if X is None:
        # if X is not specified, then generate X as random gaussian
        # if m is too large, compute X in batches
        print("entering randsphere, mem = {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        X = np.empty((m, n))
        # initial array to NAN for error checking
        X[:] = np.NAN
        batch_size = int(np.ceil(MEM_LIMIT / n))
        Nbatches = int(np.ceil(m / float(batch_size)))
        print("generating {} random vectors with dimension {}, batch size {} and we have {} batches (mem = {})".format(m, n, batch_size, Nbatches, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        for i in range(Nbatches):
            batch_start = i * batch_size
            if i == Nbatches - 1:
                batch_end = m
            else:
                batch_end = (i + 1) * batch_size
            batch_m = batch_end - batch_start
            print("batch {}/{}, start={}, end={}, batch_m = {}, mem = {}".format(i, Nbatches-1, batch_start, batch_end, batch_m, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            X_batch = np.random.randn(batch_m, n)
            s2_batch = np.sum(X_batch * X_batch, axis = 1)
            X_batch = X_batch * (np.tile(r*np.power(gammainc(n/2,s2_batch/2), 1/n) / np.sqrt(s2_batch), (n,1))).T
            X[batch_start:batch_end] = X_batch
            X_batch = None
        if np.any(np.isnan(X)):
            raise(RuntimeError("BUG: randsphere generating NaN!"))
    else:
        # for testing
        # if X is specified, then set m and n accordingly
        m = X.shape[0]
        n = X.shape[1]
        s2 = np.sum(X * X, axis = 1)
        X = X * (np.tile(r*np.power(gammainc(n/2,s2/2), 1/n) / np.sqrt(s2), (n,1))).T
    print("{} vectors generated, mem = {}".format(m, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return X
"""

if __name__ == "__main__":
    X = np.array([[0.20432, 0.74511], [0.87845, 0.52853], [0.96227, 0.69554]])
    import scipy.io as sio
    X = sio.loadmat('opt_matfiles/test.mat')['X']
    print(randsphere(3,2,0.5,X))

