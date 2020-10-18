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

How to Read Results:
After a successful run, tensorboard should write a result graph in the dir ./graph/{network name
change directory to the ./graph/lenet5 file and type in console:
                  tensorboard --logdir=./ 

Here is the result of mobilenetv2 , acc ~ 0.84 , after 30 epoches of training

![alt text](https://live.staticflickr.com/65535/50500409036_99b6a9a39c_c.jpg)
![alt text](https://live.staticflickr.com/65535/50499695193_5f048c4696_c.jpg)

To view the result of the two models , Download the following tensorboard graph: 
https://drive.google.com/file/d/1etOCcu3eAZvEH_KY_rYFKighLdfT7CIt/view?usp=sharing

Detailed explanation of the two model architecture:
Some basic knowledge abt CNN  (in Chinese though ): https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-convolution-neural-network-%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-bfa8566744e9

1. Lenet5 Architecture:
![alt text](https://miro.medium.com/max/700/0*H9_eGAtkQXJXtkoK)



2. Mobilenetv2 Architecture
![alt text](https://pic4.zhimg.com/v2-22299048d725a902a84010675fe84a13_r.jpg)
