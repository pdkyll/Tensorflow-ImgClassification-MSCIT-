import tensorflow as tf
from models.Layerprovider import LayerProvider
import src.config as config

class MobileNetv2:

    def __init__(self,shape, is4Train, mobilenetVersion=1):

        tf.reset_default_graph()# 利用这个可清空default graph以及nodes

        lProvider = LayerProvider(is4Train)
        #outputlayer = finallayerforoffsetoption()

        adaptChannels = lambda totalLayer: int(mobilenetVersion * totalLayer)

        self.inputImage = tf.placeholder(tf.float32, shape=shape, name='Image')
        #self.transtrain = opt.isTrainpre

        output = lProvider.convb(self.inputImage, 3, 3, adaptChannels(32), 2, "1-conv-32-2-1", relu=True)
        print("1-conv-32-2-1 : " + str(output.shape))

        # architecture description
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16*config.modelchannel[0], 1, 1],
            [6, 24*config.modelchannel[1], 2, 2],
            [6, 32*config.modelchannel[2], 3, 2],
            [6, 64*config.modelchannel[3], 4, 2],
            [6, 96*config.modelchannel[4], 3, 1],
            [6, 160*config.modelchannel[5], 3, 2],
            [6, 320, 1, 1]
        ]
        self.bottleneck_type = 0
        for t, c, n, s in inverted_residual_setting:
            self.bottleneck_type += 1
            output_channel = adaptChannels(c)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                layerDescription = "l" + str(self.bottleneck_type) + "-bottleneck-n" + str(i+1)
                output = lProvider.inverted_bottleneck(output, t, output_channel, stride, k_s=3, dilation=1, scope=layerDescription)

        # lProvider1 = LayerProvider(self.transtrain)
        # inverted_residual_setting1 = [
        #     # t, c, n, s
        #     [6, 320, 1, 1]
        # ]
        # for t, c, n, s in inverted_residual_setting1:
        #     self.bottleneck_type += 1
        #     output_channel = adaptChannels(c)
        #     for i in range(n):
        #         if i == 0:
        #             stride = s
        #         else:
        #             stride = 1
        #         layerDescription = "l" + str(self.bottleneck_type) + "-bottleneck-n" + str(i+1)
        #         output = lProvider1.inverted_bottleneck(output, t, output_channel, stride, k_s=3, dilation=1, scope=layerDescription)


        #for DUC

        #self.output = outputlayer.fornetworks_DUC(output,totalJoints)
        #for no DUC
        output = lProvider.convb(output, 1, 1, adaptChannels(1280), 1, "ptconv-320-1280", relu=True)
        #output = lProvider._global_avg(output, 7, 1)
        output = tf.layers.average_pooling2d(output, 7, 1,
                                    padding='valid', data_format='channels_last', name="avgpool")
        output = lProvider.convb(output, 1, 1, config.total_class, 1, "ptconv-1280-clsnum", relu=True)
        #self.output = outputlayer.fornetworks(output, config.total_class)
        output = tf.squeeze(output,[1,2],"output")

        self.output = output

    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def getInput(self):
        return self.inputImage

    def getIntermediateOutputs(self):
        return self.intermediateSupervisionOutputs[:]

    def getOutput(self):
        return self.output
