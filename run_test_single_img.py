import time
import tensorflow as tf
import numpy as np
import dataset
from datetime import timedelta
import src.config as config
from models.mobilenetv2 import MobileNetv2
from models.lenet5 import LeNet
import os
import sys
import time
import cv2
exp_dir = os.path.join("Result/")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class SingleImgTester:
    def __init__(self, model_path):

        self.model_path = model_path
        # image dimensions (only squares for now)
        self.img_size = config.img_size
        # Tuple with height and width of images used to reshape arrays.
        self.img_shape = (self.img_size, self.img_size)
        # class info
        self.num_classes = len(config.classes)
        # batch size
        self.batch_size = config.batch_size

        self.test_path = config.test_path

        self.session = tf.Session()

        with tf.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.session.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        self.session.run(tf.global_variables_initializer())
        self.x = self.session.graph.get_tensor_by_name('Image:0')
        self.y = self.session.graph.get_tensor_by_name('output:0')
        #self.x_image = tf.reshape(self.x, [-1, self.img_size, self.img_size, num_channels])

        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')

        self.y_true_cls = tf.argmax(self.y_true, axis=1)

        self.y_pred = tf.nn.softmax(self.y)

        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)

        self.test_acc = 0

        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.total_iterations = 0


    def test(self, image_path="./cat.jpg"):
            images = []
            image_normalize_mean = [0.485, 0.456, 0.406]
            image_normalize_std = [0.229, 0.224, 0.225]

            orig_image = cv2.imread(image_path)
            image = orig_image
            orig_image = cv2.resize(image, (640, 480))
            image = cv2.resize(image, (config.img_size, config.img_size))
            #cv2.imshow(image)
            images.append(image)
            images = np.array(images)
            images = images.astype(np.float32)

            images = np.multiply(images, 1.0 / 255.0)
            for image in images:
                for channel in range(3):
                    image[:, :, channel] -= image_normalize_mean[channel]
                    image[:, :, channel] /= image_normalize_std[channel]

            feed_dict_test = { self.x: images,}
            prediction = self.session.run(self.y_pred_cls, feed_dict=feed_dict_test)
            print("Prediction: ", config.classes[prediction[0]])

if __name__ == '__main__':
    T = SingleImgTester(model_path="weights/lenet5/lenet5.pb")
    T.test(image_path="./dog.jpg")