#!/usr/bin/env python3

import numpy as np
from scipy.special import gammainc


"""
Original Matlab code:

function X = randsphere(m,n,r)

X = randn(m,n);
s2 = sum(X.^2,2);
X = X.*repmat(r*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);
"""

def randsphere(m, n, r, X = None):
    if X is None:
        # if X is not specified, then generate X as random gaussian
        X = np.random.randn(m, n)
    else:
        # for testing
        # if X is specified, then set m and n accordingly
        m = X.shape[0]
        n = X.shape[1]
    s2 = np.sum(X * X, axis = 1)
    X = X * (np.tile(r*np.power(gammainc(n/2,s2/2), 1/n) / np.sqrt(s2), (n,1))).T
    return X

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

