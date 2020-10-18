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


Explanation About the Code:

How to load the batches of data to train the network?

Please Read dataset.py :
1. For reading images and labels 
2. Images path will be the fullpath of the images
3. Images will undergo Normalization and Standardization when loading the batches into the model

      a. Normalization: 1/255 --> Data will change From 0-255 to 0-1
      
      ![alt text](https://miro.medium.com/max/273/1*QWFEYIKWrBSiEqhdrGZGcA.png)
      
      b. Standardization: 
      
      ![alt text](https://miro.medium.com/max/186/1*2Nx37E6IvuITArzIs5EcCg.png)
   
              image_normalize_mean = [0.485, 0.456, 0.406]
              image_normalize_std = [0.229, 0.224, 0.225]

              for i in range(start, end):
                  image = cv2.imread(self._images_path[i])
                  image = cv2.resize(image, (self._image_size, self._image_size), cv2.INTER_LINEAR)
                  images.append(image)

              images = np.array(images)
              images = images.astype(np.float32)

              # Normalization and Standardization
              images = np.multiply(images, 1.0 / 255.0)
              for image in images:
                  for channel in range(3) :
                      image[:, :, channel] -= image_normalize_mean[channel]
                      image[:, :, channel] /= image_normalize_std[channel]
                      
Explanation in details: https://towardsdatascience.com/normalization-vs-standardization-cb8fe15082eb
   
4. labels will be in the form of [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] , where 1 represent the groundtruth of the image class and the length is 10 since there are 10 classes

Please read the trainer.py file

5. In tensorflow , we use something called session to run the input and ground truth label. To train our model we must feed in the data and let our optimizer to do back propagation and update model weighting

            x_batch, y_true_batch, _, cls_batch = self.data.train.next_batch(self.train_batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = self.data.valid.next_batch(self.train_batch_size)

            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}

            feed_dict_validate = {self.x: x_valid_batch,
                                  self.y_true: y_valid_batch}

            self.session.run(self.optimizer, feed_dict=feed_dict_train)
            
 How to define our loss and optimizer?
 
 Please read the trainer.py file
 
 1. In image classification , we usually use something called cross entrophy
  

Brief Introduction About CNN: 

Convolutional Layer extract features from the images

![alt text](https://live.staticflickr.com/65535/50500611557_cfc43b309a_z.jpg)

Pooling Layer Simplify the features and make it smaller

![alt text](https://live.staticflickr.com/65535/50500453686_678f4d7c79_z.jpg)

Activation Layer makes sure no negative values exist (Actibvation function : Linear(wont elminate -ve),  Relu, Tanh, Swish activation function (swish is a quite new act. func and it is proved to increase the acc of a network) 

![alt text](https://live.staticflickr.com/65535/50499740523_a93e59a974_z.jpg)

Some basic knowledge abt CNN  (in Chinese though ): https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-convolution-neural-network-%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-bfa8566744e9


Detailed explanation of the two model architecture: (Underconstruction)

1. Lenet5 Architecture:
![alt text](https://miro.medium.com/max/700/0*H9_eGAtkQXJXtkoK)

2. Mobilenetv2 Architecture
![alt text](https://pic4.zhimg.com/v2-22299048d725a902a84010675fe84a13_r.jpg)
