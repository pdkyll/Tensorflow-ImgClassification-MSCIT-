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
## Configuration and Hyperparameters
class Trainer:
    def __init__(self,num_channels = config.numchannels):
        # image dimensions (only squares for now)
        self.img_size = config.img_size
        # Size of image when flattened to a single dimension
        self.img_size_flat = self.img_size * self.img_size * num_channels
        # Tuple with height and width of images used to reshape arrays.
        self.img_shape = (self.img_size, self.img_size)
        # class info
        self.num_classes = len(config.classes)
        # batch size
        self.batch_size = config.batch_size
        # validation split
        self.validation_size = config.validation_size

        # how long to wait after validation loss stops improving before terminating training
        self.early_stopping = config.early_stopping

        self.train_path = config.train_path
        self.test_path = config.test_path
        self.checkpoint_dir = config.checkpoint_dir
        #tensorboard --logdir=./


        self.model = MobileNetv2((None, self.img_size, self.img_size, 3), is4Train = True, mobilenetVersion=1)
        #self.model = LeNet((None, self.img_size, self.img_size, 3))

        self.x = self.model.getInput()

        self.x_image = tf.reshape(self.x, [-1, self.img_size, self.img_size, num_channels])

        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')

        self.y_true_cls = tf.argmax(self.y_true, axis=1)

        self.y_pred = tf.nn.softmax(self.model.getOutput())

        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model.getOutput(), labels=self.y_true)

        #lr = utils.build_learning_rate(initial_lr=1e-4,global_step=1)

        self.cost = tf.reduce_mean(self.cross_entropy)

        self.train_acc = 0
        self.val_acc = 0
        self.train_loss = 0
        self.val_loss = 0

        self.epoch = 0

        self.lr = 0.01
        self.step_rate = 1000
        self.decay = 0.95
        self.time = time
        self.global_step = tf.Variable(0, trainable=False)
        #increment_global_step = tf.assign(global_step, global_step + 1)
        self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.step_rate, self.decay, staircase=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.01).minimize(self.cost, self.global_step)

        #optimizer = utils.build_optimizer(learning_rate=1e-4, optimizer_name='adam').minimize(cost)
        #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


        self.session = tf.Session()

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        self.savePath = os.path.join(exp_dir, "mobilenet2" + "checkpoints")

        #self.writer = tf.summary.FileWriter('./graphs/lenet5')
        self.writer = tf.summary.FileWriter(config.tensorboard_dir, graph_def=self.session.graph_def)

        self.train_batch_size = self.batch_size

        self.total_iterations = 0

        ## Load Data
        self.data = dataset.read_train_sets(self.train_path, self.img_size, config.classes, validation_size=self.validation_size)
        #  test_images, test_ids = dataset.read_test_set(test_path, img_size)

    def print_progress(self,epoch, feed_dict_train, feed_dict_validate, val_loss):
        # Calculate the accuracy on the training-set.
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))
        print('Learning rate: %f' % (self.session.run(self.learning_rate)), 'Global Step : %f' % (self.session.run(self.global_step)))


    def optimize(self,num_epoch):
        required_itr4_1epoch = int(self.data.train.num_examples / self.batch_size)

        # Start-time used for printing time-usage below.
        num_iterations = required_itr4_1epoch * num_epoch + 1
        start_time = time.time()

        best_val_loss = float("inf")
        patience = 0
        print("Start Training ...")
        for i in range(self.total_iterations,self.total_iterations + num_iterations):
            #lr = utils.build_learning_rate(initial_lr=lr, global_step=i)
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch, _, cls_batch = self.data.train.next_batch(self.train_batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = self.data.valid.next_batch(self.train_batch_size)

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, flattened image shape]

            #x_batch = x_batch.reshape(train_batch_size, img_size_flat)

            #x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.

            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}

            feed_dict_validate = {self.x: x_valid_batch,
                                  self.y_true: y_valid_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.

            self.session.run(self.optimizer, feed_dict=feed_dict_train)
            self.train_acc += self.session.run(self.accuracy, feed_dict=feed_dict_train)
            self.val_acc += self.session.run(self.accuracy, feed_dict=feed_dict_validate)
            self.train_loss += self.session.run(self.cost, feed_dict=feed_dict_train)
            self.val_loss += self.session.run(self.cost, feed_dict=feed_dict_validate)


            checkpoint_path = os.path.join(self.savePath, 'model')
            self.saver.save(self.session, checkpoint_path, global_step=i)

            # Print status at end of each epoch (defined as full pass through training dataset).

            self.epoch = int(i / required_itr4_1epoch)

            sys.stdout.write("\r" + "Epoch : " + str(i / required_itr4_1epoch) + " ")
            sys.stdout.flush()

            if(self.epoch != 0) :
                if (i % required_itr4_1epoch == 0) :
                    self.train_acc /= required_itr4_1epoch
                    self.val_acc /= required_itr4_1epoch
                    self.train_loss /= required_itr4_1epoch
                    self.val_loss /= required_itr4_1epoch

                    summary = tf.Summary()
                    summary.value.add(tag='train_acc', simple_value=self.train_acc)
                    summary.value.add(tag='val_acc', simple_value=self.val_acc)
                    summary.value.add(tag='train_loss', simple_value=self.train_loss)
                    summary.value.add(tag='val_loss', simple_value=self.val_loss)
                    summary.value.add(tag='learning_rate', simple_value=self.session.run(self.learning_rate))

                    self.writer.add_summary(summary, self.epoch)

                    msg = "Epoch {0} --- Training Accuracy: {1}, Validation Accuracy: {2}, Train Loss: {3}, Validation Loss: {4}"
                    print(msg.format(self.epoch, self.train_acc, self.val_acc, self.train_loss, self.val_loss))

                    #val_loss = self.session.run(self.cost, feed_dict=feed_dict_validate)
                    #epoch = int(i / int(self.data.train.num_examples / self.batch_size))
                    #self.print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

                    if self.early_stopping:
                        if self.val_loss < best_val_loss:
                            best_val_loss = self.val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience == self.early_stopping:
                            break

                    self.train_acc = 0
                    self.val_acc = 0
                    self.train_loss = 0
                    self.val_loss = 0


            # Update the total number of iterations performed.
        self.total_iterations += num_iterations

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

    def export(self):
        input_graph_def = tf.get_default_graph().as_graph_def()

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.session,
            input_graph_def,
            ["output"]
        )
        with tf.gfile.GFile("./models/mnv2.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())

        
