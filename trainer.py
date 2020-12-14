import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.model_pruning.python import pruning
import dataset
from datetime import timedelta
import src.config as config
from models.mobilenet_v2 import MobileNetv2_tf
from models.mobilenetv2 import MobileNetv2
from models.lenet5 import LeNet
from sparse import sparse_optimizers
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
        self.val_path = config.val_path
        self.test_path = config.test_path

        self.checkpoint_dir = config.checkpoint_dir

        if config.model_arch == "mobilenetv2":
            self.model = MobileNetv2((None, self.img_size, self.img_size, 3), is4Train = config.is4train, mobilenetVersion=1)

        if config.model_arch == "mobilenetv2_tf":
            self.model = MobileNetv2_tf((None, self.img_size, self.img_size, 3), is4Train = config.is4train)

        if config.model_arch == "lenet5":
            self.model = LeNet((None, self.img_size, self.img_size, 3))


        self.x = self.model.getInput()

        self.x_image = tf.reshape(self.x, [-1, self.img_size, self.img_size, num_channels])

        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')

        self.y_true_cls = tf.argmax(self.y_true, axis=1)

        #self.y_pred = tf.nn.softmax(self.model.getOutput())
        self.y_pred = self.model.getOutput()

        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model.getOutput(), labels=self.y_true)

        #lr = utils.build_learning_rate(initial_lr=1e-4,global_step=1)

        self.cost = tf.reduce_mean(self.cross_entropy)

        self.train_acc = 0
        self.val_acc = 0
        self.train_loss = 0
        self.val_loss = 0

        self.epoch = 0

        self.lr = 0.001
        self.step_rate = 1000
        self.decay = 0.95
        self.time = time

        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.step_rate, self.decay, staircase=False)
        # self.sparse_optimizer = sparse_optimizers.SparseRigLOptimizer(
        #     tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True), begin_step=0 ,
        #     end_step=25000, grow_init='zeros',
        #     frequency=100,
        #     drop_fraction=0.3,
        #     drop_fraction_anneal='constant',
        #     initial_acc_scale= 0., use_tpu=False).minimize(self.cost,self.global_step)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,self.global_step)

        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Get, Print, and Edit Pruning Hyperparameters
        pruning_hparams = pruning.get_pruning_hparams()
        print("Pruning Hyperparameters:", pruning_hparams)

        # Change hyperparameters to meet our needs
        pruning_hparams.begin_pruning_step = 100
        #pruning_hparams.end_pruning_step = 250
        pruning_hparams.sparsity_function_end_step = 10000
        pruning_hparams.pruning_frequency = 100
        #pruning_hparams.initial_sparsity= .3
        pruning_hparams.target_sparsity = .9

        # Create a pruning object using the pruning specification, sparsity seems to have priority over the hparam
        p = pruning.Pruning(pruning_hparams, global_step=self.global_step)
        self.prune_op = p.conditional_mask_update_op()

        self.session = tf.Session()
        # self.saver = tf.train.Saver(max_to_keep=10)
        # self.savePath = os.path.join(exp_dir, "mobilenet2" + "checkpoints")
        self.session.run(tf.global_variables_initializer())
        if config.is4train == True:
            #tf.reset_default_graph()
            self.saver = tf.train.Saver(max_to_keep=10)
            self.savePath = os.path.join(exp_dir, config.model_arch + "checkpoints")
        else:

            self.saver = tf.train.import_meta_graph(config.pretrained_checkpoint_dir + '.meta')
            self.saver.restore(self.session, config.pretrained_checkpoint_dir)
            #self.session.run(tf.global_variables_initializer())

        for variable in slim.get_variables():
            tf.summary.histogram(variable.op.name, variable)

        self.summaries = tf.summary.merge_all()

        #self.writer = tf.summary.FileWriter('./graphs/lenet5')
        self.train_writer = tf.summary.FileWriter(config.tensorboard_dir + "/plot_train", graph_def=self.session.graph_def)
        self.val_writer = tf.summary.FileWriter(config.tensorboard_dir + "/plot_val", graph_def=self.session.graph_def)
        self.train_batch_size = self.batch_size

        self.total_iterations = 0

        ## Load Data for Training and Validation
        self.data = dataset.read_train_sets(self.train_path, self.val_path, self.img_size, config.classes)
        #  test_images, test_ids = dataset.read_test_set(test_path, img_size)

    def print_progress(self,epoch, feed_dict_train, feed_dict_validate, val_loss):
        # Calculate the accuracy on the training-set.
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))
        print('Learning rate: %f' % (self.session.run(self.learning_rate)), 'Global Step : %f' % (self.session.run(self.global_step)))

    def restore(self, checkpointPath):
        saver = tf.train.import_meta_graph(checkpointPath + '.meta')
        saver.restore(self.session, checkpointPath)
        #tf.train.Saver().restore(self.session, checkpointPath)

    def optimize(self,num_epoch):
        required_itr4_1epoch = int(self.data.train.num_examples / self.batch_size)

        # Start-time used for printing time-usage below.
        num_iterations = required_itr4_1epoch * num_epoch + 1

        if config.is4oneitr == True:
            num_iterations = 1

        start_time = time.time()

        best_val_acc = 0
        pre_val_acc = 0
        current_val_acc = 0
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

            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}

            feed_dict_validate = {self.x: x_valid_batch,
                                  self.y_true: y_valid_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.

            self.session.run(self.optimizer, feed_dict=feed_dict_train)
            #self.session.run(self.sparse_optimizer, feed_dict=feed_dict_train)
            self.train_acc += self.session.run(self.accuracy, feed_dict=feed_dict_train)
            self.val_acc += self.session.run(self.accuracy, feed_dict=feed_dict_validate)
            self.train_loss += self.session.run(self.cost, feed_dict=feed_dict_train)
            self.val_loss += self.session.run(self.cost, feed_dict=feed_dict_validate)

            # summ = self.session.run(self.summaries,feed_dict=feed_dict_validate)

            self.epoch = int(i / required_itr4_1epoch)

            sys.stdout.write("\r" + "Epoch : " + str(i / required_itr4_1epoch) + " ")
            sys.stdout.flush()

            if(self.epoch != 0) :
                if (i % required_itr4_1epoch == 0) :
                    self.train_acc /= required_itr4_1epoch
                    self.val_acc /= required_itr4_1epoch
                    self.train_loss /= required_itr4_1epoch
                    self.val_loss /= required_itr4_1epoch

                    train_summary = tf.Summary()
                    val_summary = tf.Summary()

                    train_summary.value.add(tag='acc', simple_value=self.train_acc)
                    val_summary.value.add(tag='acc', simple_value=self.val_acc)

                    train_summary.value.add(tag='loss', simple_value=self.train_loss)
                    val_summary.value.add(tag='loss', simple_value=self.val_loss)

                    train_summary.value.add(tag='learning_rate', simple_value=self.session.run(self.learning_rate))

                    self.train_writer.add_summary(self.session.run(self.summaries), self.epoch)

                    self.train_writer.add_summary(train_summary, self.epoch)
                    self.val_writer.add_summary(val_summary, self.epoch)


                    msg = "Epoch {0} --- Training Accuracy: {1}, Validation Accuracy: {2}, Train Loss: {3}, Validation Loss: {4}"
                    print(msg.format(self.epoch, self.train_acc, self.val_acc, self.train_loss, self.val_loss))

                    if not os.path.exists(config.checkpoint_dir):
                        os.makedirs(config.checkpoint_dir)

                    if config.is4train == True:
                        self.saver.save(self.session, config.checkpoint_dir + config.model_arch , global_step=i)

                    current_val_acc = self.val_acc

                    if best_val_acc < current_val_acc and current_val_acc > 0.5:
                        best_val_acc = current_val_acc
                        self.export(name= "acc="+str(current_val_acc))

                    #print("Sparsity of layers (should be 0)", self.session.run(tf.contrib.model_pruning.get_weight_sparsity()))

                    self.train_acc = 0
                    self.val_acc = 0
                    self.train_loss = 0
                    self.val_loss = 0

            if config.is4oneitr == True:
                self.export(name="_training=false")

        self.total_iterations += num_iterations

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

    def prune(self,num_epoch):

        self.total_iterations = 0

        self.session.run(tf.assign(self.global_step, 0))

        required_itr4_1epoch = int(self.data.train.num_examples / self.batch_size)

        # Start-time used for printing time-usage below.
        num_iterations = required_itr4_1epoch * num_epoch + 1

        if config.is4oneitr == True:
            num_iterations = 1

        start_time = time.time()

        best_val_acc = 0

        print("Start Pruning ...")
        for i in range(self.total_iterations,self.total_iterations + num_iterations):
            self.session.run(self.prune_op)
            x_batch, y_true_batch, _, cls_batch = self.data.train.next_batch(self.train_batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = self.data.valid.next_batch(self.train_batch_size)

            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}

            feed_dict_validate = {self.x: x_valid_batch,
                                  self.y_true: y_valid_batch}

            #self.session.run(self.optimizer, feed_dict=feed_dict_train)
            self.session.run(self.sparse_optimizer, feed_dict=feed_dict_train)
            self.train_acc += self.session.run(self.accuracy, feed_dict=feed_dict_train)
            self.val_acc += self.session.run(self.accuracy, feed_dict=feed_dict_validate)
            self.train_loss += self.session.run(self.cost, feed_dict=feed_dict_train)
            self.val_loss += self.session.run(self.cost, feed_dict=feed_dict_validate)


            self.epoch = int(i / required_itr4_1epoch)

            sys.stdout.write("\r" + "Epoch : " + str(i / required_itr4_1epoch) + " ")
            sys.stdout.flush()

            if(self.epoch != 0) :
                if (i % required_itr4_1epoch == 0) :
                    self.train_acc /= required_itr4_1epoch
                    self.val_acc /= required_itr4_1epoch
                    self.train_loss /= required_itr4_1epoch
                    self.val_loss /= required_itr4_1epoch

                    # train_summary = tf.Summary()
                    # val_summary = tf.Summary()
                    #
                    # train_summary.value.add(tag='acc', simple_value=self.train_acc)
                    # val_summary.value.add(tag='acc', simple_value=self.val_acc)
                    #
                    # train_summary.value.add(tag='loss', simple_value=self.train_loss)
                    # val_summary.value.add(tag='loss', simple_value=self.val_loss)
                    #
                    # train_summary.value.add(tag='learning_rate', simple_value=self.session.run(self.learning_rate))
                    #
                    # self.train_writer.add_summary(self.session.run(self.summaries), self.epoch)
                    #
                    # self.train_writer.add_summary(train_summary, self.epoch)
                    # self.val_writer.add_summary(val_summary, self.epoch)


                    msg = "Epoch {0} --- Training Accuracy: {1}, Validation Accuracy: {2}, Train Loss: {3}, Validation Loss: {4}"
                    print(msg.format(self.epoch, self.train_acc, self.val_acc, self.train_loss, self.val_loss))

                    # if not os.path.exists(config.checkpoint_dir):
                    #     os.makedirs(config.checkpoint_dir)
                    #
                    # if config.is4train == True:
                    #     self.saver.save(self.session, config.checkpoint_dir + config.model_arch , global_step=i)

                    current_val_acc = self.val_acc

                    if best_val_acc < current_val_acc and current_val_acc > 0.1:
                        best_val_acc = current_val_acc
                        self.export(name= "_pruned_acc="+str(current_val_acc))

                    self.train_acc = 0
                    self.val_acc = 0
                    self.train_loss = 0
                    self.val_loss = 0

                    print("Sparsity of layers (should be 0)",
                          self.session.run(tf.contrib.model_pruning.get_weight_sparsity()))
                    mask_vals = self.session.run(pruning.get_masks())


        self.total_iterations += num_iterations



        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

    def export(self, name):
        input_graph_def = tf.get_default_graph().as_graph_def()

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.session,
            input_graph_def,
            ["output"]
        )
        if not os.path.exists(config.weight_dir):
            os.makedirs(config.weight_dir)

        with tf.gfile.GFile(config.weight_dir+ config.model_arch + str(name) +".pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())

        
