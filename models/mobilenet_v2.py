import tensorflow as tf
import src.config as config
from models.ops import *

class MobileNetv2_tf:
    def __init__(self, shape, is4Train, exp=6):

        is_train = is4Train
        self.inputImage = tf.placeholder(tf.float32, shape=shape, name='Image')

        net = conv2d_block(self.inputImage, 32, 3, 2, is_train, name='conv1_1')  # size/2

        net = res_block(net, 1, 16, 1, is_train, name='res2_1')

        net = res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
        net = res_block(net, exp, 24, 1, is_train, name='res3_2')

        net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
        net = res_block(net, exp, 32, 1, is_train, name='res4_2')
        net = res_block(net, exp, 32, 1, is_train, name='res4_3')

        net = res_block(net, exp, 64, 2, is_train, name='res5_1')
        net = res_block(net, exp, 64, 1, is_train, name='res5_2')
        net = res_block(net, exp, 64, 1, is_train, name='res5_3')
        net = res_block(net, exp, 64, 1, is_train, name='res5_4')

        net = res_block(net, exp, 96, 1, is_train, name='res6_1')  # size/16
        net = res_block(net, exp, 96, 1, is_train, name='res6_2')
        net = res_block(net, exp, 96, 1, is_train, name='res6_3')

        net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
        net = res_block(net, exp, 160, 1, is_train, name='res7_2')
        net = res_block(net, exp, 160, 1, is_train, name='res7_3')

        net = res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)

        net = pwise_block(net, 1280, is_train, name='conv9_1')
        net = global_avg(net)
        output = flatten(conv_1x1(net, config.total_class, name='o'))
        output = tf.identity(output, "output")
        self.output = output

    def getInput(self):
        return self.inputImage

    def getOutput(self):
        return self.output
