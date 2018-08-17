
"""
train_models.py

train the neural network models for attacking

Copyright (C) 2017-2018, IBM Corp.
Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>

This program is licenced under the Apache 2.0 licence,
contained in the LICENCE file in this directory.
"""

import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import SGD, Adam
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import argparse
import os

# train nlayer MLP models
def train(data, file_name, params, num_epochs=50, batch_size=256, train_temp=1, init=None, lr=0.01, decay=1e-5, momentum=0.9, activation="relu", optimizer_name="sgd"):
    """
    Train a n-layer simple network for MNIST and CIFAR
    """
    
    # create a Keras sequential model
    model = Sequential()
    # reshape the input (28*28*1) or (32*32*3) to 1-D
    model.add(Flatten(input_shape=data.train_data.shape[1:]))
    # dense layers (the hidden layer)
    n = 0
    for param in params:
        n += 1
        model.add(Dense(param, kernel_initializer='he_uniform'))
        # ReLU activation
        if activation == "arctan":
            model.add(Lambda(lambda x: tf.atan(x), name=activation+"_"+str(n)))
        else:
            model.add(Activation(activation, name=activation+"_"+str(n)))
    # the output layer, with 10 classes
    model.add(Dense(10, kernel_initializer='he_uniform'))
    
    # load initial weights when given
    if init != None:
        model.load_weights(init)

    # define the loss function which is the cross entropy between prediction and true label
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    if optimizer_name == "sgd":
        # initiate the SGD optimizer with given hyper parameters
        optimizer = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    elif optimizer_name == "adam":
        optimizer = Adam(lr=lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay=decay, amsgrad=False)

    # compile the Keras model, given the specified loss and optimizer
    model.compile(loss=fn,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    model.summary()
    print("Traing a {} layer model, saving to {}".format(len(params) + 1, file_name))
    # run training with given dataset, and print progress
    history = model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name)
        print('model saved to ', file_name)
    
    return {'model':model, 'history':history}

# train cnn 7-layer mnist/cifar model
def train_cnn_7layer(data, file_name, params, num_epochs=50, batch_size=256, train_temp=1, init=None, lr=0.01, decay=1e-5, momentum=0.9, activation="relu", optimizer_name="sgd"):
    """
    Train a 7-layer cnn network for MNIST and CIFAR (same as the cnn model in Clever)
    mnist: 32 32 64 64 200 200 
    cifar: 64 64 128 128 256 256
    """

    # create a Keras sequential model
    model = Sequential()

    print("training data shape = {}".format(data.train_data.shape))

    # define model structure
    model.add(Conv2D(params[0], (3, 3),
                            input_shape=data.train_data.shape[1:]))
    model.add(Activation(activation))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation(activation))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation(activation))
    model.add(Dense(10))

  
    # load initial weights when given
    if init != None:
        model.load_weights(init)

    # define the loss function which is the cross entropy between prediction and true label
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    if optimizer_name == "sgd":
        # initiate the SGD optimizer with given hyper parameters
        optimizer = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    elif optimizer_name == "adam":
        optimizer = Adam(lr=lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay=decay, amsgrad=False)
    
    # compile the Keras model, given the specified loss and optimizer
    model.compile(loss=fn,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    model.summary()
    print("Traing a {} layer model, saving to {}".format(len(params) + 1, file_name))
    # run training with given dataset, and print progress
    history = model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name)
        print('model saved to ', file_name)
    
    return {'model':model, 'history':history}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train n-layer MNIST and CIFAR MLP or CNN 7-layer models')
    parser.add_argument('--model', 
                default="mnist",
                choices=["mnist", "cifar"],
                help='model name')
    parser.add_argument('--modelfile', 
                default="",
                help='override the model filename, use user specied one')
    parser.add_argument('--modelpath', 
                default="models_training",
                help='folder for saving trained models')
    parser.add_argument('--modeltype', 
                default="mlp",
                choices=["mlp", "cnn"],
                help='model type')
    parser.add_argument('layer_parameters',
                nargs='+',
                help='number of hidden units per layer')
    parser.add_argument('--activation',
                default="relu",
                choices=["relu", "tanh", "sigmoid", "arctan", "elu", "hard_sigmoid", "softplus"])
    parser.add_argument('--lr',
                default=-1,
                type=float,
                help='learning rate')
    parser.add_argument('--wd',
                default=-1,
                type=float,
                help='weight decay')
    parser.add_argument('--epochs',
                default=50,
                type=int,
                help='number of epochs')
    parser.add_argument('--optimizer', 
                default="sgd",
                choices=["sgd", "adam"],
                help='optimizer type')
    parser.add_argument('--overwrite',
                action='store_true',
                help='overwrite output file')
    args = parser.parse_args()
    print(args)
    nlayers = len(args.layer_parameters) + 1
    if not args.modelfile:
        if args.modeltype == "mlp":
            file_name = args.modelpath+"/"+args.model+"_"+args.modeltype+"_"+str(nlayers)+"layer_"+args.activation+"_"+args.layer_parameters[0]
        elif args.modeltype == "cnn":
            file_name = args.modelpath+"/"+args.model+"_"+args.modeltype+"_"+str(nlayers)+"layer_"+args.activation
            
    else:
        file_name = args.modelfile
    print("Model will be saved to", file_name)
    if os.path.isfile(file_name) and not args.overwrite:
        raise RuntimeError("model {} exists.".format(file_name))
    # load data
    if args.model == "mnist":
        data = MNIST()
    elif args.model == "cifar":
        data = CIFAR()
    
    # save trained model path
    if not os.path.isdir(args.modelpath):
        os.makedirs(args.modelpath)
    # set optimizer parameters
    if args.lr == -1:
        if args.optimizer == "sgd":
            args.lr = 0.01 # default value
        elif args.optimizer == "adam":
            args.lr = 0.001

    if args.wd == -1:
        if args.optimizer == "sgd":
            args.wd = 1e-5 # default value
        elif args.optimizer == "adam":
            args.wd = 0
        
    # start training:
    if args.modeltype == "mlp":
        train(data, file_name=file_name, params=args.layer_parameters, num_epochs=args.epochs, lr=args.lr, decay=args.wd, activation=args.activation, optimizer_name=args.optimizer)
    elif args.modeltype == "cnn":
        train_cnn_7layer(data, file_name=file_name, params=args.layer_parameters, num_epochs=args.epochs, lr=args.lr, decay=args.wd, activation=args.activation, optimizer_name=args.optimizer)
   

    # 2-layer models
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[10], num_epochs=50, lr=0.03, decay=1e-6)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[50], num_epochs=50, lr=0.05,decay=1e-4)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[100], num_epochs=50, lr=0.05, decay=1e-4)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[1024], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_2layer_relu", params=[1024], num_epochs=50, lr=0.2, decay=1e-3)
    # 3-layer models
    # train(MNIST(), file_name="models/mnist_3layer_relu", params=[10, 10], num_epochs=50, lr=0.03, decay=1e-7)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[50], num_epochs=50, lr=0.05,decay=1e-4)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[100], num_epochs=50, lr=0.05, decay=1e-4)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[1024], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_2layer_relu", params=[1024], num_epochs=50, lr=0.2, decay=1e-3)
    # 3-layer models
    # train(MNIST(), file_name="models/mnist_3layer_relu", params=[10, 10], num_epochs=50, lr=0.03, decay=1e-7)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[100], num_epochs=50, lr=0.05, decay=1e-4)

    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[1024], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_2layer_relu", params=[1024], num_epochs=50, lr=0.2, decay=1e-3)
    # 3-layer models
    # train(MNIST(), file_name="models/mnist_3layer_relu_10_10", params=[10, 10], num_epochs=50, lr=0.03, decay=1e-7)
    # train(MNIST(), file_name="models/mnist_3layer_relu", params=[256,256], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_3layer_relu", params=[256,256], num_epochs=50, lr=0.2, decay=1e-3)
    # 4-layer models
    # train(MNIST(), file_name="models/mnist_4layer_relu", params=[256,256,256], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_4layer_relu", params=[256,256,256], num_epochs=50, lr=0.2, decay=1e-3)
    # train(MNIST(), file_name="models/mnist_4layer_relu", params=[20,20,20], num_epochs=50, lr=0.07, decay=1e-3)
    # train(MNIST(), file_name="models/mnist_5layer_relu", params=[20,20,20,20], num_epochs=50, lr=0.03, decay=1e-4) 
    # train(MNIST(), file_name="models/mnist_5layer_relu", params=[20,20,20,20], num_epochs=50, lr=0.02, decay=1e-4)

