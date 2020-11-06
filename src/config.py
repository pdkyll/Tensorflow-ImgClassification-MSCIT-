from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%Y/%m/%d/%H:%M:%S")
device = "cuda:0"

train_path = 'data/cifar10png/train'
val_path = 'data/cifar10png/val'
test_path = 'data/cifar10png/test'

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
total_class = len(classes)

model_arch = "mobilenetv2"

if model_arch == "mobilenetv2":
    img_size = 224

if model_arch == "lenet5":
    img_size = 32

is4train = False
is4oneitr = True
pretrained_checkpoint_dir = "./checkpoints/mobilenetv2/2020/11/06/16:27:35/mobilenetv2-1093"

# is4train = True
# is4oneitr = False
# pretrained_checkpoint_dir = None


checkpoint_dir = "./checkpoints/" + model_arch + "/" + date_time + "/"
weight_dir = "./weights/" + model_arch + "/" + date_time + "/"
tensorboard_dir = './graphs/' + model_arch + "/" + date_time

numchannels = 3
# batch size
batch_size = 32
# validation split
validation_size = .16

# use None if you don't want to implement early stopping
early_stopping = False

#mobilenet v2 modelchannel
modelchannel = [1,1,1,1,1,1]
