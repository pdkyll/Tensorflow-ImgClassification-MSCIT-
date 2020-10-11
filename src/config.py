
device = "cuda:0"

train_path = 'data/cifar10png/train'
test_path = 'data/catdog/test'
checkpoint_dir = "models/"

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
total_class = len(classes)
img_size = 224

# batch size
batch_size = 32
# validation split
validation_size = .16

# use None if you don't want to implement early stopping
early_stopping = False

#mobilenet v2 modelchannel
modelchannel = [1,1,1,1,1,1]

