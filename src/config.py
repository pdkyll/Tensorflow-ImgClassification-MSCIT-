from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%Y/%m/%d/%H-%M-%S")
device = "cuda:0"

train_path = r'C:\Users\hkuit164\Desktop\CNN_classification\data\CatDog\train'
val_path = r'C:\Users\hkuit164\Desktop\CNN_classification\data\CatDog\val'
test_path = r'C:\Users\hkuit164\Desktop\CNN_classification\data\CatDog\val'

classes = ['cat', "dog"]
total_class = len(classes)

model_arch = "mobilenetv2"

if model_arch == "mobilenetv2" or "mobilenetv2_tf":
    img_size = 224

if model_arch == "lenet5":
    img_size = 32

is4train = False
is4oneitr = True
pretrained_checkpoint_dir = "./checkpoints/mobilenetv2/2020/11/23/17-21-06/mobilenetv2-5460"

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
