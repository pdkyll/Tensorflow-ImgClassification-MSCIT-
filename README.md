# Tensorflow-ImgClassification-MSCIT-
Dataset:

CIFAR-10 Dataset
Download Link: 
https://drive.google.com/file/d/1XpBBM95ZhlF_QuDc54rLQalbQ1aDfv-m/view?usp=sharing

Unzip the file and create a dir name "data" , put "cifar10png" inside

Environment's Requirement:

- Python == 3.6.9
- numpy == 1.16.5 (anaconda* / pip)
- tensorflow-gpu == 1.14 (anaconda* / pip)
- tensorboard == 1.14 (anaconda* / pip)
- opencv-python == 4.1.1.26 (pip*)
- sklearn == 0.0 (pip*)

* is just my choice in installing those packages

Folder Dir Arch:

      Lcheckpoints
      Lconvert 
      Ldata
        Lcifar10png
      Lgraphs
      Lmodels
      Lsrc 
        Lconfig.py
      Lutils
      Lweights
      dataset.py
      trainer.py
      run.py 

How to Run:

1. execute run.py 

    T = Trainer()
    T.optimize(num_epoch=1)
    T.export()

Option: Select number of epoch for training

(Optional) 2. Edit Config in src/config.py to select different models  
i. select different models
      model_arch = "lenet5"

      if model_arch == "mobilenetv2":
          img_size = 224

      if model_arch == "lenet5":
          img_size = 32

Option: The current supporting models include lenet5 , mobilenetv2

ii. change validation size

      # batch size
      batch_size = 32
      # validation split
      validation_size = .16

Option: The default bs is 32 and train : valid spilt is 84 : 16 
