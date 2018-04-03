#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_gradients.py

Front end for collecting maximum gradient norm samples

Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>
"""

from __future__ import division

import numpy as np
import scipy.io as sio
import random
import time
import sys
import os
from functools import partial

from estimate_gradient_norm import EstimateLipschitz
from utils import generate_data

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description = "Collect gradient norm samples for calculating CLEVER score", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar", "imagenet"], default="mnist", help = "choose dataset")
    parser.add_argument("-m", "--model_name", default="normal", 
            help = "Select model. For MNIST and CIFAR, options are: 2-layer (MLP), normal (7-layer CNN), distilled (7-layer CNN with defensive distillation), brelu (7-layer CNN with Bounded ReLU). For ImageNet, available options are: resnet_v2_50, resnet_v2_101, resnet_v2_152, inception_v1, inception_v2, inception_v3, inception_v4, inception_resnet_v2, vgg_16, vgg_19, mobilenet_v1_025, mobilenet_v1_050, mobilenet_v1_100, alexnet, nasnet_large, densenet121_k32, densenet169_k32, densenet161_k48")
    parser.add_argument("-N", "--Nsamps", type=int, default=1024, help = "number of samples per iterations")
    parser.add_argument("-i", "--Niters", type=int, default=500, help = "number of iterations. NITERS maximum gradient norms will be collected. A larger value will give a more accurate estimate")
    parser.add_argument("-n", "--numimg", type=int, default=1, help = "number of test images to load from dataset")
    parser.add_argument("--ids", default = "", help = "use a filelist of image IDs in CSV file for attack (UNSUPPORTED)")
    parser.add_argument("--target_type", type=int, default=0b01111, help = "Binary mask for selecting targeted attack classes. bit0: top-2, bit1: random, bit2: least likely, bit3: use --ids override (UNSUPPORTED), bit4: use all labels (for untargeted)")
    parser.add_argument("-f", "--firstimg", type=int, default=0, help = "start from which image in dataset")
    parser.add_argument("--compute_slope", action='store_true', help = "collect slope estimate")
    parser.add_argument("--sample_norm", type=str, choices=["l2", "li", "l1"], help = "norm of sampling ball (l2, l1 or li)", default="l2")
    parser.add_argument("--fix_dirty_bug", action='store_true', help = "do not use (UNSUPPORTED)")
    parser.add_argument("-b", "--batch_size", type=int, default=0, help = "batch size to run model. 0: use default batch size")
    parser.add_argument("--nthreads", type=int, default=0, help = "number of threads for generating random samples in sphere")
    parser.add_argument("-s", "--save", default="./lipschitz_mat", help = "results output path")
    parser.add_argument("--seed", type=int, default = 1215, help = "random seed")
    
    args = vars(parser.parse_args())
    print(args)
    
    seed = args['seed']
    Nsamp = args['Nsamps'];
    Niters = args['Niters'];
    dataset = args['dataset']
    model_name = args['model_name']
    start = args['firstimg']
    numimg = args['numimg']
    save_path = args['save']
    total = 0

    random.seed(seed)
    np.random.seed(seed)

    # create output directory
    os.system("mkdir -p {}/{}_{}".format(save_path, dataset, model_name))

    # create a Lipschitz estimator class (initial it early to save multiprocessing memory)
    clever_estimator = EstimateLipschitz(sess=None, nthreads=args['nthreads'])

    # import the ID lists
    if args['ids']:
        import pandas as pd
        df = pd.read_csv(args['ids'], sep = "\t")
        # don't use this
        if args['fix_dirty_bug']:
            df = df[df['new_class'] != df['target']]
        ids = list(df['id'])
        target_type = args['target_type']
        # use the target classes override
        if target_type & 0b1000 != 0:
            target_classes = [[t] for t in list(df['target'])]
        else:
            # use generated classes
            target_classes = None
    else:
        ids = None
        target_classes = None
        target_type = args['target_type']

    import tensorflow as tf
    from setup_cifar import CIFAR
    from setup_mnist import MNIST
    from setup_imagenet import ImageNet

    tf.set_random_seed(seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        clever_estimator.sess = sess
        # returns the input tensor and output prediction vector
        img, output = clever_estimator.load_model(dataset, model_name, batch_size = args['batch_size'], compute_slope = args['compute_slope'])
        # load dataset
        datasets_loader = {"mnist": MNIST, "cifar": CIFAR, "imagenet": partial(ImageNet, clever_estimator.model.image_size)}
        data = datasets_loader[dataset]()
        # for prediction
        predictor = lambda x: np.squeeze(sess.run(output, feed_dict = {img: x}))
        # generate target images
        inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=numimg, targeted=True,
                                    start=start, predictor=predictor,
                                    random_and_least_likely = True,
                                    ids = ids, target_classes = target_classes, target_type = target_type,
                                    imagenet="imagenet" in dataset,
                                    remove_background_class="imagenet" in dataset and 
                                    ("vgg" in model_name or "densenet" in model_name or "alexnet" in model_name))

        timestart = time.time()
        print("got {} images".format(inputs.shape))
        for i, input_img in enumerate(inputs):
            # original_predict = np.squeeze(sess.run(output, feed_dict = {img: [input_img]}))
            print("processing image {}".format(i))
            original_predict = predictor([input_img])
            true_label = np.argmax(true_labels[i])
            predicted_label = np.argmax(original_predict)
            least_likely_label = np.argmin(original_predict)
            original_prob = np.sort(original_predict)
            original_class = np.argsort(original_predict)
            print("Top-10 classifications:", original_class[-1:-11:-1])
            print("True label:", true_label)
            print("Top-10 probabilities/logits:", original_prob[-1:-11:-1])
            print("Most unlikely classifications:", original_class[:10])
            print("Most unlikely probabilities/logits:", original_prob[:10])
            if true_label != predicted_label:
                print("[WARNING] This image is classfied wrongly by the classifier! Skipping!")
                continue
            total += 1
            # set target class
            target_label = np.argmax(targets[i]);
            print('Target class: ', target_label)
            sys.stdout.flush()
            
            [L2_max,L1_max,Li_max,G2_max,G1_max,Gi_max,g_x0,pred] = clever_estimator.estimate(input_img, true_label, target_label, Nsamp, Niters, args['sample_norm'])
            print("[STATS][L1] total = {}, seq = {}, id = {}, time = {:.3f}, true_class = {}, target_class = {}, info = {}".format(total, i, true_ids[i], time.time() - timestart, true_label, target_label, img_info[i]))
            # save to sampling results to matlab ;)
            mat_path = "{}/{}_{}/{}_{}_{}_{}_{}_{}".format(save_path, dataset, model_name, Nsamp, Niters, true_ids[i], true_label, target_label, img_info[i])
            save_dict = {'L2_max': L2_max, 'L1_max': L1_max, 'Li_max': Li_max, 'G2_max': G2_max, 'G1_max': G1_max, 'Gi_max': Gi_max, 'pred': pred, 'g_x0': g_x0, 'id': true_ids[i], 'true_label': true_label, 'target_label': target_label, 'info':img_info[i], 'args': args, 'path': mat_path}
            sio.savemat(mat_path, save_dict)
            print('saved to', mat_path)
            sys.stdout.flush()

