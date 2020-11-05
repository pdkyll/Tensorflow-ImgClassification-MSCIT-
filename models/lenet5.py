import tensorflow as tf
from tensorflow.contrib.layers import flatten
import src.config as config

class LeNet:

    def __init__(self , shape):
        # Hyperparameters
        mu = 0
        sigma = 0.1
        layer_depth = {
            'layer_2': 16,
            'layer_3': 120,
            'layer_f1': 84
        }

        self.inputImage = tf.placeholder(tf.float32, shape=shape, name='Image')

        he_norm_initializer = tf.contrib.layers.variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
        #x = tf.identity(x, "Input")
        # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        # conv1_w = tf.get_variable("conv1_w", shape=(5, 5, 3, 6), initializer=he_norm_initializer)
        # conv1_b = tf.Variable(tf.zeros(6))
        conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 6], mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(6))

        conv1 = tf.nn.conv2d(self.inputImage, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)

        # TODO: Activation.
        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        # conv2_w = tf.get_variable("conv2_w", shape=(5, 5, 6, 16), initializer=he_norm_initializer)
        # conv2_b = tf.Variable(tf.zeros(16))
        conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(16))

        conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        # TODO: Activation.
        conv2 = tf.nn.relu(conv2)

        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        init = tf.constant_initializer(0.1)
        # TODO: Flatten. Input = 5x5x16. Output = 400.
        fc1 = flatten(pool_2)

        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_w = tf.get_variable("fc1_w", shape=(400, 120), initializer=he_norm_initializer)
        fc1_b = tf.get_variable("fc1_b", shape=(120), initializer=he_norm_initializer)

        #fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
        #fc1_b = tf.Variable(init(shape=[120], dtype=tf.float32))
        fc1 = tf.matmul(fc1, fc1_w) + fc1_b

        # TODO: Activation.
        fc1 = tf.nn.relu(fc1)

        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_w = tf.get_variable("fc2_w", shape=(120, 84), initializer=he_norm_initializer)
        fc2_b = tf.get_variable("fc2_b", shape=(84), initializer=he_norm_initializer)

        #fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
        #fc2_b = tf.Variable(init(shape=[84], dtype=tf.float32))

        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        # TODO: Activation.
        fc2 = tf.nn.relu(fc2)

        # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
        # fc3_w = tf.get_variable("fc3_w", shape=(84, config.total_class), initializer=he_norm_initializer)
        # fc3_b = tf.get_variable("fc3_b", shape=[config.total_class], initializer=he_norm_initializer)

        fc3_w = tf.Variable(tf.truncated_normal(shape=(84, config.total_class), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(init(shape=[config.total_class], dtype=tf.float32))

        output = tf.matmul(fc2, fc3_w) + fc3_b
        output = tf.identity(output,"output")
        self.output = output

    def getInput(self):
        return self.inputImage

    def getOutput(self):
        return self.output