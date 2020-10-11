import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta

import matplotlib
import src.config as config
from models.mobilenetv2 import MobileNetv2
import utils.utils as utils

## Configuration and Hyperparameters

num_channels = 3
# image dimensions (only squares for now)
img_size = config.img_size
# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# class info
classes = config.classes
num_classes = len(classes)
# batch size
batch_size = config.batch_size
# validation split
validation_size = config.validation_size

# how long to wait after validation loss stops improving before terminating training
early_stopping = config.early_stopping

train_path = config.train_path
test_path = config.test_path
checkpoint_dir = config.checkpoint_dir
#tensorboard --logdir=./


model = MobileNetv2((None, img_size, img_size, 3), is4Train = True, mobilenetVersion=1)

x = model.getInput()

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, axis=1)

y_pred = tf.nn.softmax(model.getOutput())

y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.getOutput(), labels=y_true)

#lr = utils.build_learning_rate(initial_lr=1e-4,global_step=1)

cost = tf.reduce_mean(cross_entropy)

lr = 0.01
step_rate = 1000
decay = 0.95

global_step = tf.Variable(0, trainable=False)
increment_global_step = tf.assign(global_step, global_step + 1)
learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.01).minimize(cost, global_step)

#optimizer = utils.build_optimizer(learning_rate=1e-4, optimizer_name='adam').minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = batch_size

total_iterations = 0

## Load Data
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
#  test_images, test_ids = dataset.read_test_set(test_path, img_size)

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
    print('Learning rate: %f' % (session.run(learning_rate)))

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,total_iterations + num_iterations):
        #lr = utils.build_learning_rate(initial_lr=lr, global_step=i)
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        #x_batch = x_batch.reshape(train_batch_size, img_size_flat)

        #x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples / batch_size))

            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


optimize(num_iterations=1000000)