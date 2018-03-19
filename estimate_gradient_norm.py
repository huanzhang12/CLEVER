#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_gradient_norm.py

A multithreaded gradient norm sampler

Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>
"""

from __future__ import division

import numpy as np
import random
import ctypes
import time
import sys
import os

from multiprocessing import Pool, current_process, cpu_count
from shmemarray import ShmemRawArray, NpShmemArray
from functools import partial
from randsphere import randsphere

     
class EstimateLipschitz(object):

    def __init__(self, sess, seed = 1215, nthreads = 0):
        """
        sess: tensorflow session
        Nsamp: number of samples to take per iteration
        Niters: number of iterations, each iteration we return a max L
        """
        self.sess = sess
        self.seed = seed
        # create a pool of workers to compute samples in advance
        if nthreads == 0:
            self.n_processes = max(cpu_count() // 2, 1)
        else:
            self.n_processes = nthreads
        # set up random seed during initialization
        self.pool = Pool(processes = self.n_processes, initializer = lambda s: np.random.seed(s + current_process()._identity[0]), initargs=(self.seed,))

    def load_model(self, dataset = "mnist", model_name = "2-layer", model = None, batch_size = 0, compute_slope = False):
        """
        model: if set to None, then load dataset with model_name. Otherwise use the model directly.
        dataset: mnist, cifar and imagenet. recommend to use mnist and cifar as a starting point.
        model_name: possible options are 2-layer, distilled, and normal
        """
        import tensorflow as tf
        from setup_cifar import CIFAR, CIFARModel, TwoLayerCIFARModel
        from setup_mnist import MNIST, MNISTModel, TwoLayerMNISTModel
        from nlayer_model import NLayerModel
        from setup_imagenet import ImageNet, ImageNetModel

        # if set this to true, we will use the logit layer output instead of probability
        # the logit layer's gradients are usually larger and more stable
        output_logits = True
        self.dataset = dataset
        self.model_name = model_name

        if model is None:
            print('Loading model...')
            if dataset == "mnist":
                self.batch_size = 1024
                if model_name == "2-layer":
                    model = TwoLayerMNISTModel("models/mnist_2layer", self.sess, not output_logits)
                elif model_name == "normal":
                    model =  MNISTModel("models/mnist", self.sess, not output_logits)
                elif model_name == "brelu":
                    model =  MNISTModel("models/mnist_brelu", self.sess, not output_logits, use_brelu = True)
                elif model_name == "distilled":
                    model =  MNISTModel("models/mnist-distilled-100", self.sess, not output_logits)
                else:
                    # specify model parameters as N,M,opts
                    model_params = model_name.split(",")
                    if len(model_params) < 3:
                        raise(RuntimeError("incorrect model option" + model_name))
                    numlayer = int(model_params[0])
                    nhidden = int(model_params[1])
                    modelfile = "models/mnist_{}layer_relu_{}_{}".format(numlayer, nhidden, model_params[2])
                    print("loading", modelfile)
                    model = NLayerModel([nhidden] * (numlayer - 1), modelfile)
            elif dataset == "cifar":
                self.batch_size = 1024
                if model_name == "2-layer":
                    model = TwoLayerCIFARModel("models/cifar_2layer", self.sess, not output_logits)
                elif model_name == "normal":
                    model = CIFARModel("models/cifar", self.sess, not output_logits)
                elif model_name == "brelu":
                    model = CIFARModel("models/cifar_brelu", self.sess, not output_logits, use_brelu = True)
                elif model_name == "distilled":
                    model = CIFARModel("models/cifar-distilled-100", self.sess, not output_logits)
                else:
                    # specify model parameters as N,M,opts
                    model_params = model_name.split(",")
                    if len(model_params) < 3:
                        raise(RuntimeError("incorrect model option" + model_name))
                    numlayer = int(model_params[0])
                    nhidden = int(model_params[1])
                    modelfile = "models/cifar_{}layer_relu_{}_{}".format(numlayer, nhidden, model_params[2])
                    print("loading", modelfile)
                    model = NLayerModel([nhidden] * (numlayer - 1), modelfile, image_size=32, image_channel=3)
            elif dataset == "imagenet":
                self.batch_size = 32
                model = ImageNetModel(self.sess, use_softmax = not output_logits, model_name = model_name, create_prediction = False)
            else:
                raise(RuntimeError("dataset unknown"))

        self.model = model
        self.compute_slope = compute_slope
        if batch_size != 0:
            self.batch_size = batch_size

        # img is the placeholder for image input
        self.img = tf.placeholder(shape = [None, model.image_size, model.image_size, model.num_channels], dtype = tf.float32)
        # output is the output tensor of the entire network
        self.output = model.predict(self.img)
        # create the graph to compute gradient
        # get the desired true label and target label
        self.true_label = tf.placeholder(dtype = tf.int32, shape = [])
        self.target_label = tf.placeholder(dtype = tf.int32, shape = [])
        true_output = self.output[:, self.true_label]
        target_output = self.output[:, self.target_label]
        # get the different
        self.objective = true_output - target_output
        # get the gradient
        self.grad_op = tf.gradients(self.objective, self.img)[0]
        # compute gradient norm
        grad_op_rs = tf.reshape(self.grad_op, (tf.shape(self.grad_op)[0], -1))
        self.grad_2_norm_op = tf.norm(grad_op_rs, axis = 1)
        self.grad_1_norm_op = tf.norm(grad_op_rs, ord=1, axis = 1)
        self.grad_inf_norm_op = tf.norm(grad_op_rs, ord=np.inf, axis = 1)

        return self.img, self.output

    def _estimate_Lipschitz_multiplerun(self, num, niters, input_image, target_label, true_label):
        """
        num: number of samples per iteration
        niters: number of iterations
        input_image: original image (h*w*c)
        """
        batch_size = self.batch_size
        shape = (batch_size, self.model.image_size, self.model.image_size, self.model.num_channels)
        dimension = self.model.image_size * self.model.image_size * self.model.num_channels

        if num < batch_size:
            print("Increasing num to", batch_size)
            num = batch_size

        # get the original prediction
        pred, grad_val, grad_2_norm_val, grad_1_norm_val, grad_inf_norm_val = self.sess.run(
          [self.output, self.grad_op, self.grad_2_norm_op, self.grad_1_norm_op, self.grad_inf_norm_op], 
          feed_dict = {self.img: [input_image], self.true_label: true_label, self.target_label: target_label})
        pred = np.squeeze(pred)
        # class c and class j in Hein's paper. c is original class
        c = true_label
        j = target_label
        # get g_x0 = f_c(x_0) - f_j(x_0)
        g_x0 = pred[c] - pred[j]
        print('g_x0 = '+ str(g_x0))
        # print(pred)
        # print(grad_val)
        print(grad_2_norm_val, grad_1_norm_val, grad_inf_norm_val)

        if self.dataset == "imagenet":
            # for imagenet, generate random samples for this batch only
            item_per_process = int(np.ceil(batch_size / float(self.n_processes)))
            last_process_item = batch_size - item_per_process * (self.n_processes - 1)
            # array in shared memory storing results of all threads
            total_item_size = batch_size
        else:
            # for cifar and mnist, generate random samples for this entire iteration
            item_per_process = int(np.ceil(num / float(self.n_processes)))
            last_process_item = num - item_per_process * (self.n_processes - 1)
            total_item_size = num
        # array in shared memory storing results of all threads
        """
        shared_data = ShmemRawArray('f', total_item_size * dimension, tag)
        result_arr = np.ctypeslib.as_array(shared_data)
        result_arr = result_arr.reshape(total_item_size, dimension)
        """
        # create necessary shared array structures
        inputs_0 = np.array(input_image)
        tag_prefix = str(os.getpid()) + "_"
        result_arr = NpShmemArray(np.float32, (total_item_size, dimension), tag_prefix + "randsphere")
        # we have an extra batch_size to avoid overflow
        # the scaling constant in [a,b]: scale the L2 norm of each sample (has originally norm ~1)
        a = 0; b = 3; 
        scale = NpShmemArray(np.float32, (num+batch_size), tag_prefix + "scale")
        scale[:] = (b-a)*np.random.rand(num+batch_size)+a; 
        input_example = NpShmemArray(np.float32, inputs_0.shape, tag_prefix + "input_example")
        # this is a read-only array
        input_example[:] = inputs_0
        # all_inputs holds the perturbations for one batch or all samples
        all_inputs = NpShmemArray(np.float32, (total_item_size,) + inputs_0.shape, tag_prefix + "all_inputs")
        # prepare the argument list
        process_item_list = (self.n_processes - 1) * [item_per_process] + [last_process_item]
        offset_list = [0]
        for item in process_item_list[:-1]:
            offset_list.append(offset_list[-1] + item)
        print(self.n_processes, "threads launched with paramter", process_item_list, offset_list)

        # create multiple threads to generate samples
        worker_func = partial(randsphere, n = dimension, input_shape = inputs_0.shape, total_size = total_item_size, scale_size = num+batch_size, tag_prefix = tag_prefix, r = 1.0, X = None)
        worker_args = list(zip(process_item_list, offset_list, [0] * self.n_processes))
        sample_results = self.pool.map_async(worker_func, worker_args)

        # num: # of samples to be run, \leq samples.shape[0]
        
        # number of iterations
        Niters = niters;
        # store the max L in each iteration
        L2_max = np.zeros(Niters)
        L1_max = np.zeros(Niters)
        Li_max = np.zeros(Niters)
        # store the max G in each iteration
        G2_max = np.zeros(Niters)
        G1_max = np.zeros(Niters)
        Gi_max = np.zeros(Niters)
        # store computed Lispschitz constants in each iteration
        L2 = np.zeros(num)
        L1 = np.zeros(num)
        Li = np.zeros(num)
        # store computed gradient norm in each iteration
        G2 = np.zeros(num)
        G1 = np.zeros(num)
        Gi = np.zeros(num)
        # how many batches we have
        Nbatches = num // batch_size

        # timer
        search_begin_time = time.time()

        # multiple runs: shuffling the samples 
        for iters in range(Niters):
            
            # shuffled index 
            # idx_shuffle = np.random.permutation(num);
            
            # the scaling constant in [a,b]: scale the L2 norm of each sample (has originally norm ~1)
            scale[:] = (b-a)*np.random.rand(num+batch_size)+a; 

            # number of L's we have computed
            L_counter = 0
            G_counter = 0

            # for cifar and mnist, generate random samples for this entire iteration
            if self.dataset != "imagenet":
                # get samples for this iteration
                sample_results.get()
                # create multiple threads to generate samples for next batch
                sample_results = self.pool.map_async(worker_func, worker_args)

            overhead_time = 0.0
            for i in range(Nbatches):
                overhead_start = time.time()
                # for imagenet, generate random samples for this batch only
                if self.dataset == "imagenet":
                    # get samples for this batch
                    sample_results.get()
                    # create multiple threads to generate samples for next batch
                    worker_args = zip(process_item_list, offset_list, [(i + 1) * batch_size] * self.n_processes)
                    sample_results = self.pool.map_async(worker_func, worker_args)

                """
                # gather all indices
                ii = list(range(i * batch_size, (i + 1) * batch_size))
                # element wise product
                if self.dataset == "imagenet":
                    # s = scale[ii][:, np.newaxis] * samples
                    s = scale[i * batch_size: (i + 1) * batch_size][:, np.newaxis] * samples
                else:
                    # s = scale[ii][:, np.newaxis] * samples[ii, :]
                    s = scale[i * batch_size: (i + 1) * batch_size][:, np.newaxis] * samples[i * batch_size: (i + 1) * batch_size, :]
                # prepare batched inputs
                batch_inputs = np.repeat(inputs_0, batch_size, axis = 0)
                batch_inputs += np.reshape(s, shape)
                # clip the input to valid image range
                """
                if self.dataset == "imagenet":
                    # we generate samples for each batch at a time
                    batch_inputs = all_inputs
                else:
                    # we generate samples for all batches
                    batch_inputs = all_inputs[i * batch_size: (i + 1) * batch_size]
                np.clip(batch_inputs, -0.5, 0.5, out = batch_inputs)
                overhead_time += time.time() - overhead_start

                # run inference and get the gradient
                perturbed_predicts, perturbed_grad_2_norm, perturbed_grad_1_norm, perturbed_grad_inf_norm = self.sess.run(
                        [self.output, self.grad_2_norm_op, self.grad_1_norm_op, self.grad_inf_norm_op], 
                        feed_dict = {self.img: batch_inputs, self.target_label: target_label, self.true_label: true_label})
                
                overhead_start = time.time()
                if self.compute_slope:
                    # compute distance between consecutive samples: not use sequential samples 
                    s12_2_norm = np.linalg.norm(s[0:batch_size-1:2] - s[1:batch_size:2], axis = 1)
                    s12_1_norm = np.linalg.norm(s[0:batch_size-1:2] - s[1:batch_size:2], ord=1, axis = 1)
                    s12_i_norm = np.linalg.norm(s[0:batch_size-1:2] - s[1:batch_size:2], ord=np.inf, axis = 1)
                    # compute function value differences: not use sequential samples 
                    g_x1 = perturbed_predicts[0:batch_size-1:2, c] - perturbed_predicts[0:batch_size-1:2, j]
                    g_x2 = perturbed_predicts[1:batch_size:2, c] - perturbed_predicts[1:batch_size:2, j]
                    # estimated Lipschitz constants for this batch
                    # for slope estimate, we need the DUAL norm
                    batch_L2 = np.abs(g_x1 - g_x2) / s12_2_norm
                    batch_L1 = np.abs(g_x1 - g_x2) / s12_i_norm
                    batch_Li = np.abs(g_x1 - g_x2) / s12_1_norm
                    L2[L_counter : L_counter + batch_size//2] = batch_L2
                    L1[L_counter : L_counter + batch_size//2] = batch_L1
                    Li[L_counter : L_counter + batch_size//2] = batch_Li
                G2[G_counter : G_counter + batch_size] = perturbed_grad_2_norm
                G1[G_counter : G_counter + batch_size] = perturbed_grad_1_norm
                Gi[G_counter : G_counter + batch_size] = perturbed_grad_inf_norm
                L_counter += (batch_size//2)
                G_counter += batch_size
                overhead_time += time.time() - overhead_start
            
            # get the per-iteration max gradient norm
            if self.compute_slope:
                L2_max[iters] = np.max(L2)
                L1_max[iters] = np.max(L1)
                Li_max[iters] = np.max(Li)
            G2_max[iters] = np.max(G2)
            G1_max[iters] = np.max(G1)
            Gi_max[iters] = np.max(Gi)
            print('[STATS][L2] loop = {}, time = {:.5g}, overhead = {:.5g}, L2 = {:.5g}, L1 = {:.5g}, Linf = {:.5g}, G2 = {:.5g}, G1 = {:.5g}, Ginf = {:.5g}'.format(iters, time.time() - search_begin_time, overhead_time, L2_max[iters], L1_max[iters], Li_max[iters], G2_max[iters], G1_max[iters], Gi_max[iters]))
            sys.stdout.flush()
            # reset per iteration L and G
            if self.compute_slope:
                L2.fill(0)
                L1.fill(0)
                Li.fill(0)
            G2.fill(0)
            G1.fill(0)
            Gi.fill(0)
            
        print('[STATS][L1] g_x0 = {:.5g}, L2_max = {:.5g}, L1_max = {:.5g}, Linf_max = {:.5g}, G2_max = {:.5g}, G1_max = {:.5g}, Ginf_max = {:.5g}'.format(
               g_x0, np.max(L2_max), np.max(L1_max), np.max(Li_max), np.max(G2_max), np.max(G1_max), np.max(Gi_max)))
        # when compute the bound we need the DUAL norm
        if self.compute_slope:
            print('[STATS][L1] bnd_L2_max = {:.5g}, bnd_L1_max = {:.5g}, bnd_Linf_max = {:.5g}, bnd_G2_max = {:.5g}, bnd_G1_max = {:.5g}, bnd_Ginf_max = {:.5g}'.format(g_x0/np.max(L2_max), g_x0/np.max(Li_max), g_x0/np.max(L1_max), g_x0/np.max(G2_max), g_x0/np.max(Gi_max), g_x0/np.max(G1_max)))
        else:
            print('[STATS][L1] bnd_G2_max = {:.5g}, bnd_G1_max = {:.5g}, bnd_Ginf_max = {:.5g}'.format(g_x0/np.max(G2_max), g_x0/np.max(Gi_max), g_x0/np.max(G1_max)))
        
        sys.stdout.flush()

        # discard the last batch of samples
        sample_results.get()
        return [L2_max,L1_max,Li_max,G2_max,G1_max,Gi_max,g_x0,pred]

    def __del__(self):
        # terminate the pool
        self.pool.terminate()

    def estimate(self, x_0, true_label, target_label, Nsamp, Niters):
        result = self._estimate_Lipschitz_multiplerun(Nsamp,Niters,x_0,target_label,true_label)
        return result


