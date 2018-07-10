#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## train_2layer.py -- train MLP models for MNIST and CIFAR
##
## Copyright (C) 2017-2018, IBM Corp.
## Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
##                and Huan Zhang <ecezhang@ucdavis.edu>
##
## This program is licenced under the Apache 2.0 licence,
## contained in the LICENCE file in this directory.


import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os

def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None, lr=0.01, decay=1e-5, momentum=0.9):
    """
    Train a 2-layer simple network for MNIST and CIFAR
    """
    # create a Keras sequential model
    model = Sequential()
    # reshape the input (28*28*1) or (32*32*3) to 1-D
    model.add(Flatten(input_shape=data.train_data.shape[1:]))
    # first dense layer (the hidden layer)
    model.add(Dense(params[0]))
    # \alpha = 10 in softplus, multiply input by 10
    model.add(Lambda(lambda x: x * 10))
    # in Keras the softplus activation cannot set \alpha
    model.add(Activation('softplus'))
    # so manually add \alpha to the network
    model.add(Lambda(lambda x: x * 0.1))
    # the output layer, with 10 classes
    model.add(Dense(10))
    
    # load initial weights when given
    if init != None:
        model.load_weights(init)

    # define the loss function which is the cross entropy between prediction and true label
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    # initiate the SGD optimizer with given hyper parameters
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    
    # compile the Keras model, given the specified loss and optimizer
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    # run training with given dataset, and print progress
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name)

    return model

if not os.path.isdir('models'):
    os.makedirs('models')

if __name__ == "__main__":
    train(MNIST(), file_name="models/mnist_2layer", params=[1024], num_epochs=50, lr=0.1, decay=1e-3)
    train(CIFAR(), file_name="models/cifar_2layer", params=[1024], num_epochs=50, lr=0.2, decay=1e-3)

