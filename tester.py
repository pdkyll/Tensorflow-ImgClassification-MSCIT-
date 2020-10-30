import time
import tensorflow as tf
import dataset
from datetime import timedelta
import src.config as config
from models.mobilenetv2 import MobileNetv2
from models.lenet5 import LeNet
import os
import sys
import time
exp_dir = os.path.join("Result/")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Tester:
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

        self.data = dataset.read_test_set(self.test_path, self.img_size, config.classes)



    def test(self):
        num_epoch = 1
        required_itr4_1epoch = int(self.data.test.num_examples / self.batch_size)
        num_iterations = required_itr4_1epoch * num_epoch + 1

        for i in range(self.total_iterations, self.total_iterations + num_iterations):
            x_batch, y_true_batch, _, cls_batch = self.data.test.next_batch(self.batch_size)
            feed_dict_test = { self.x: x_batch,
                               self.y_true: y_true_batch }
            self.test_acc += self.session.run(self.accuracy, feed_dict=feed_dict_test)
            if i % 100 == 0:
                msg = "{0} % of the dataset is tested"
                completeness = (i/num_iterations)*100
                print(msg.format(completeness))

        self.test_acc /= required_itr4_1epoch

        print("The Final Accuracy is : " , self.test_acc)