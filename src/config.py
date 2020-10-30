device = "cuda:0"

train_path = 'data/cifar10png/train'
val_path = 'data/cifar10png/val'
test_path = 'data/cifar10png/test'

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
total_class = len(classes)

model_arch = "lenet5"

if model_arch == "mobilenetv2":
    img_size = 224

if model_arch == "lenet5":
    img_size = 32

checkpoint_dir = "./checkpoints/" + model_arch + "/"
weight_dir = "./weights/" + model_arch + "/"
tensorboard_dir = './graphs/' + model_arch

numchannels = 3
# batch size
batch_size = 32
# validation split
validation_size = .16

# use None if you don't want to implement early stopping
early_stopping = False

#mobilenet v2 modelchannel
modelchannel = [1,1,1,1,1,1]
