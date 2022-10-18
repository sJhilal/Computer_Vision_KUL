#!/usr/bin/env python
# coding: utf-8

# <div style="width:100%; height:140px">
#     <img src="https://www.kuleuven.be/internationaal/thinktank/fotos-en-logos/ku-leuven-logo.png/image_preview" width = 300px, heigh = auto align=left>
# </div>
# 
# 
# KUL H02A5a Computer Vision: Group Assignment 2
# ---------------------------------------------------------------
# 
# In this group assignment your team will delve into some deep learning applications for computer vision. The assignment will be delivered in the same groups from *Group assignment 1* and you start from this template notebook. The notebook you submit for grading is the last notebook you submit in the [Kaggle competition](https://www.kaggle.com/t/d11be6a431b84198bc85f54ae7e2563f) prior to the deadline on **Tuesday 24 May 23:59**. Closely follow [these instructions](https://github.com/gourie/kaggle_inclass) for joining the competition, sharing your notebook with the TAs and making a valid notebook submission to the competition. A notebook submission not only produces a *submission.csv* file that is used to calculate your competition score, it also runs the entire notebook and saves its output as if it were a report. This way it becomes an all-in-one-place document for the TAs to review. As such, please make sure that your final submission notebook is self-contained and fully documented (e.g. provide strong arguments for the design choices that you make). Most likely, this notebook format is not appropriate to run all your experiments at submission time (e.g. the training of CNNs is a memory hungry and time consuming process; due to limited Kaggle resources). It can be a good idea to distribute your code otherwise and only summarize your findings, together with your final predictions, in the submission notebook. For example, you can substitute experiments with some text and figures that you have produced "offline" (e.g. learning curves and results on your internal validation set or even the test set for different architectures, pre-processing pipelines, etc). We advise you to first go through the PDF of this assignment entirely before you really start. Then, it can be a good idea to go through this notebook and use it as your first notebook submission to the competition. You can make use of the *Group assignment 2* forum/discussion board on Toledo if you have any questions. Good luck and have fun!
# 
# ---------------------------------------------------------------
# 

# # 1. Overview
# This assignment consists of *three main parts* for which we expect you to provide code and extensive documentation in the notebook:
# * Image classification (Sect. 2)
# * Semantic segmentation (Sect. 3)
# * Adversarial attacks (Sect. 4)
# 
# In the first part, you will train an end-to-end neural network for image classification. In the second part, you will do the same for semantic segmentation. For these two tasks we expect you to put a significant effort into optimizing performance and as such competing with fellow students via the Kaggle competition. In the third part, you will try to find and exploit the weaknesses of your classification and/or segmentation network. For the latter there is no competition format, but we do expect you to put significant effort in achieving good performance on the self-posed goal for that part. Finally, we ask you to reflect and produce an overall discussion with links to the lectures and "real world" computer vision (Sect. 5). It is important to note that only a small part of the grade will reflect the actual performance of your networks. However, we do expect all things to work! In general, we will evaluate the correctness of your approach and your understanding of what you have done that you demonstrate in the descriptions and discussions in the final notebook.

# ## 1.1 Deep learning resources
# If you did not yet explore this in *Group assignment 1 (Sect. 2)*, we recommend using the TensorFlow and/or Keras library for building deep learning models. You can find a nice crash course [here](https://colab.research.google.com/drive/1UCJt8EYjlzCs1H1d1X0iDGYJsHKwu-NO).

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
import numpy as np
np.random.seed(42)
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv3D,MaxPooling2D,Dense,Flatten,Dropout, Input, GlobalAveragePooling2D
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import cv2
import os, sys
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import HammingLoss
from tensorflow.random import set_seed
import keras.backend as K
from glob import glob
from PIL import Image
set_seed(42)

import warnings


# ## 1.2 PASCAL VOC 2009
# For this project you will be using the [PASCAL VOC 2009](http://host.robots.ox.ac.uk/pascal/VOC/voc2009/index.html) dataset. This dataset consists of colour images of various scenes with different object classes (e.g. animal: *bird, cat, ...*; vehicle: *aeroplane, bicycle, ...*), totalling 20 classes.

# In[2]:


# Loading the training data
train_df = pd.read_csv('/kaggle/input/kul-h02a5a-computer-vision-ga2-2022/train/train_set.csv', index_col="Id")
labels = train_df.columns
train_df["img"] = [np.load('/kaggle/input/kul-h02a5a-computer-vision-ga2-2022/train/img/train_{}.npy'.format(idx)) for idx, _ in train_df.iterrows()]
train_df["seg"] = [np.load('/kaggle/input/kul-h02a5a-computer-vision-ga2-2022/train/seg/train_{}.npy'.format(idx)) for idx, _ in train_df.iterrows()]
print("The training set contains {} examples.".format(len(train_df)))

# Show some examples
fig, axs = plt.subplots(2, 20, figsize=(10 * 20, 10 * 2))
for i, label in enumerate(labels):
    df = train_df.loc[train_df[label] == 1]
    axs[0, i].imshow(df.iloc[0]["img"], vmin=0, vmax=255)
    axs[0, i].set_title("\n".join(label for label in labels if df.iloc[0][label] == 1), fontsize=40)
    axs[0, i].axis("off")
    axs[1, i].imshow(df.iloc[0]["seg"], vmin=0, vmax=20)  # with the absolute color scale it will be clear that the arrays in the "seg" column are label maps (labels in [0, 20])
    axs[1, i].axis("off")
    
plt.show()

# The training dataframe contains for each image 20 columns with the ground truth classification labels and 20 column with the ground truth segmentation maps for each class
train_df.head(1)


# In[3]:


# Loading the test data
test_df = pd.read_csv('/kaggle/input/kul-h02a5a-computer-vision-ga2-2022/test/test_set.csv', index_col="Id")
test_df["img"] = [np.load('/kaggle/input/kul-h02a5a-computer-vision-ga2-2022/test/img/test_{}.npy'.format(idx)) for idx, _ in test_df.iterrows()]
test_df["seg"] = [-1 * np.ones(img.shape[:2], dtype=np.int8) for img in test_df["img"]]
print("The test set contains {} examples.".format(len(test_df)))

# The test dataframe is similar to the training dataframe, but here the values are -1 --> your task is to fill in these as good as possible in Sect. 2 and Sect. 3; in Sect. 6 this dataframe is automatically transformed in the submission CSV!
test_df.head(1)


# ## 1.3 Your Kaggle submission
# Your filled test dataframe (during Sect. 2 and Sect. 3) must be converted to a submission.csv with two rows per example (one for classification and one for segmentation) and with only a single prediction column (the multi-class/label predictions running length encoded). You don't need to edit this section. Just make sure to call this function at the right position in this notebook.

# In[4]:


def _rle_encode(img):
    """
    Kaggle requires RLE encoded predictions for computation of the Dice score (https://www.kaggle.com/lifa08/run-length-encode-and-decode)

    Parameters
    ----------
    img: np.ndarray - binary img array
    
    Returns
    -------
    rle: String - running length encoded version of img
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle

def generate_submission(df):
    """
    Make sure to call this function once after you completed Sect. 2 and Sect. 3! It transforms and writes your test dataframe into a submission.csv file.
    
    Parameters
    ----------
    df: pd.DataFrame - filled dataframe that needs to be converted
    
    Returns
    -------
    submission_df: pd.DataFrame - df in submission format.
    """
    df_dict = {"Id": [], "Predicted": []}
    for idx, _ in df.iterrows():
        df_dict["Id"].append(f"{idx}_classification")
        df_dict["Predicted"].append(_rle_encode(np.array(df.loc[idx, labels])))
        df_dict["Id"].append(f"{idx}_segmentation")
        df_dict["Predicted"].append(_rle_encode(np.array([df.loc[idx, "seg"] == j + 1 for j in range(len(labels))])))
    
    submission_df = pd.DataFrame(data=df_dict, dtype=str).set_index("Id")
    submission_df.to_csv("submission.csv")
    return submission_df


# # 2. Image classification
# 
# The goal here is simple: implement a classification CNN and train it to recognise all 20 classes (and/or background) using the training set and compete on the test set (by filling in the classification columns in the test dataframe).

# ## 2.1 CNN From Scratch
#  
# In this first section, we will create a model that will only be trained on the training data provided (749 training images for 20 classes). The amount of data given for training is small, thus the neural networks possibility to learn from the examples given is limited. This has two effects: Firstly, the algorithm is not perfectly trained on the training data, second, the model’s ability to predict the classes on the basis of the test set is therefore also limited. Since we have few examples, our number one concern should be overfitting. Overfitting happens when a model exposed to too few examples learns patterns that do not generalise to new data, i.e. when the model starts using irrelevant features for making predictions.
#  
# In order to prevent the latter and produce good outcomes, several techniques can be applied:
# Firstly, data augmentation is one way to fight overfitting. By doing so, batches of tensor image data with real-time data augmentation are generated. The augmentation operations are rotation, rescaling, turning, shearing, scale changes and flips. Yet this is not sufficient since our augmented samples are still highly correlated.
#  
# Secondly, overfitting can be remedied by increasing the information a model can store and convey. By generating and calculating more features, a model can store more information, but it is also more at risk to start storing irrelevant features. As a consequence, the choice of the number of parameters in the model, i.e. the number of layers and the size of each layer is important.
#  
# The first method of creating a model will be from scratch. For this the Keras library will be used. We're however only given 749 training images for 20 classes, which is definitely not a lot and probably not enough to learn this network from scratch.

# ### 2.1.1 Data resizing and augmentation
#  
# Below, we create two functions. One function to resize all images into the required size as per model for the transfer learning (such as 224, 224 for ResNet50V2).
# The second function implements data augmentation: In order to overcome the issue of the small amount of training data that we have, we will use data augmentation.
#  
# Image data augmentation is a technique that can be used to expand the size of a training dataset by creating alternative versions of each image in the dataset. These alternative versions of the image are created by applying different edits to the images (e.g. zoom, brightness, flipping... etc) which not only expands the training data, but also makes the model more robust to variations in the images that it will have to classify in the test data.
# 
# 
# 
# 

# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

def img_resize(img_df, img_size):
    resized_img = np.empty((len(img_df), img_size[0], img_size[1], 3))
    i = 0
    for img in img_df["img"]:
        img = tf.image.resize_with_pad(img, img_size[0], img_size[1])
        resized_img[i] = img
        i += 1
    return resized_img

def train_data_augmentation(img_df, img_size, batch):
    resized_img = img_resize(img_df, img_size)
    
    Aug_gen = ImageDataGenerator(
        rotation_range=20,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    train_data_generator = Aug_gen.flow(
            x = resized_img,
            y = img_df.iloc[:, :20].to_numpy(),
            batch_size= batch)
    
    return train_data_generator

def test_data_augmentation(img_df, img_size, batch):
    resized_img = img_resize(img_df, img_size)
    
    Aug_gen = ImageDataGenerator(rescale=1./255)
    
    test_data_generator = Aug_gen.flow(
            x = resized_img,
            y = None,
            batch_size= batch)
    
    return test_data_generator


# ### 2.1.2 Class distibution and weight adjustments
# 
# First, let's check the number of examples we have for each class in our training data.

# In[6]:


hits = []
for label in train_df[labels]:
    hits.append(np.count_nonzero(train_df[label] == 1))

print(hits)


# We seem to have a relatively even distribtion for all the classes other than the element 15 (person) which has 207 examples in our training data. This might induce some bias in our model as it will lean into predicting the class 15 more than the other classes just because of the distribution of our training data. 
# 
# In order to account for this bias and solve this problem, we will introduce class weights in our model. The code below will be used to calculate the weights for each class.

# In[7]:


from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.preprocessing import LabelEncoder
# Create a pd.series that represents the categorical class of each one-hot encoded row
y_classes = train_df[labels].idxmax(1, skipna=False)

# Instantiate the label encoder
lbl_enc = LabelEncoder()

# Fit the label encoder to our label series
lbl_enc.fit(list(y_classes))

# Create integer based labels Series
y_integers = lbl_enc.transform(list(y_classes))

# Create dict of labels : integer representation
labels_and_integers = dict(zip(y_classes, y_integers))
class_weights = compute_class_weight(class_weight ='balanced', classes = np.unique(y_integers), y = y_integers)
sample_weights = compute_sample_weight('balanced', y_integers)
class_weights_dict = dict(zip(lbl_enc.transform(list(lbl_enc.classes_)), class_weights))


# ### 2.1.3 CNN model
# 
# We are using a CNN model based on the AlexNet architecture. The model consists of:
# 
# - 5 convolutional layers: Convolutions allows the extraction of usefel informations such as edges in each step. Each convolutional layer is followed by a hidden relu activation layer. Of these 5 layers, 3 are also followed by a pooling layer (in order to decrease the size of the input and the computational costs) with batch-normalization layers between each activation adn pooling layers.
# 
# - 3 fully-connected layers: These layers connec all neurons of one layer with all neurons of a second allowing the model to keep the context of the whole image into account when predicting a class. After all, an image is only recognisable because of the combination, one edge on itself doesn't say much. Each layer is followed by a hidden relu activation layer in addition to one dropout layer to avoid overfitting. 
# 
# Alexnet was constructed with images of size 227x227x3. Since we are only replicating the Alexnet architecture but not using a pretrained model, the image size is not bound to 227x227x3. In the following, we applied a shape of 150x150x. 

# In[8]:


from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization

batch_size = 16
image_size = (150, 150)
image_shape= (image_size[0], image_size[1], 3)

CNN_model = Sequential()

# Convolutional Layers
CNN_model.add(Conv2D(64, (7, 7), input_shape= image_shape))
CNN_model.add(Activation('relu'))
CNN_model.add(BatchNormalization())
CNN_model.add(MaxPooling2D(pool_size=(3, 3)))

CNN_model.add(Conv2D(128, (5, 5)))
CNN_model.add(Activation('relu'))
CNN_model.add(BatchNormalization())
CNN_model.add(MaxPooling2D(pool_size=(3, 3)))
        
CNN_model.add(Conv2D(145, (3, 3)))
CNN_model.add(Activation('relu'))

CNN_model.add(Conv2D(145, (3, 3)))
CNN_model.add(Activation('relu'))
        
CNN_model.add(Conv2D(128, (3, 3)))
CNN_model.add(Activation('relu'))
CNN_model.add(BatchNormalization())
CNN_model.add(MaxPooling2D(pool_size=(4, 4)))

# Fully connected layers
CNN_model.add(Flatten())

CNN_model.add(Dense(64))
CNN_model.add(Activation('relu'))

CNN_model.add(Dense(64))
CNN_model.add(Activation('relu'))
CNN_model.add(Dropout(0.5))

CNN_model.add(Dense(20))
CNN_model.add(Activation('sigmoid'))

CNN_model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics= 'accuracy')

# Model summary 
CNN_model.summary()


# We will train our CNN model on the augmentated data (using the class weights)

# In[9]:


train_generator = train_data_augmentation(train_df, image_size, batch_size)

# Code line below if we want to train this model

CNN_model.fit(train_generator, steps_per_epoch = 749 // batch_size, class_weight = class_weights_dict, epochs = 150, shuffle = True)


# From the self-implemented CNN model based on AlexNet architecture we can see the following: Training is slow, the accuracy is at about 0.66 after 150 epochs, the model loss is continuously reducing. However, given the small dataset, the training gains are small and we risk overfitting on the training set. 
# 

# ## 2.2 Transfer Learning
# 
# Our CNN model trained from scratch is not performing well mainly due to the ammount of training data that we have even after we augmented the data. In order to have a better accuracy we need a larger ammount of training data which we do not have. An alternative solution is to use transfer learning, by taking a model that has already been trained on a huge collection of data and then retrain the final learning on our training data.
# 
# We experimented with three models: VGG16, ResNet50V2 and MobileNetV2. The models differ by their size and design of layers and the data they have been trained on. 
# 
# The choice of the different models & datasets for transfer learning will be explained later on. Yet it has an influence on the performance of the algorithm: 
# 
# If the source data for the pretrained model is similar to the target model, overfitting is a potential problem. If the source model and source data are greatly differing from each other, achieving the target model can take a longer time. 
# 
# All three models can be found below. However, the one that performed best for us and the one we will be using in our predictions is MobileNetV2.

# ### 2.2.1 VGG16
# 

# In the following, the model VGG16 is used to pretrain the network on the VGG16 dataset. The VGG16 network by Karen Simonyan and Andrew Zisserman of the Visual Geometry Group Lab of Oxford University in 2014 in the paper “Very deep convolutional networks for large-scale image recognition”. The inputs of VGG16 are of fixed size of 224x224 and have RGB channels. The corresponding tensor is 440x (224,224,3) and the output is 440x26 – thus a classification value for each class. Keras provides further options to pre-process the images before feeding them which transforms RBG to BGR channels and each colour channel is zero-centered with respect to the ImageNet dataset, without scaling.
# 
# Data as prepared as follows: 1) all images were resized to the required shape of 224,224,3, all input categories were reshaped to a one-hot encoded array for each of the 440 input images. The tensors were then fed into the model. The input to the network is image of dimensions (224, 224, 3). The VGG16 network combines several layers with different channel and filter size. 
# The following layers have then the following dimensions: 64 channels of 3x3 filter, same padding, max pool layer of stride (2, 2), two layers with convolution of 256 filter size and filter size (3, 3), max pooling layer of stride (2, 2), 2 convolution layers of filter size (3, 3) and 256 filter, followed by 2 sets of 3 convolution layer and a max pool layer. Each have 512 filters of (3, 3) size with same padding, then two convolution and max pooling layers. 
# The filter is a 3x3 kernel in comparison to 11x11 in AlexNet. A 1x1 pixel filter is used to manipulate the number of input channels. A padding of 1-pixel (same padding) done after each convolution layer is done to maintain the spatial features of the image. To make sure the probabilities on the predicted y values add up to 1, a softmax function was implemented. 
# 
# The last layers on top of the VGG16 network are then working on the PASCAL VOC dataset. The model was not trained on the PASCAL VOC data but did not provide satisfactory results.  
# 
# ### Evaluation
# 
# The current model still suffers overfitting: the training error is low, but the validation error does not reduce. This is not very satisfactory, as it points towards the model fitting well on the training data but not on the test data, thus the models capability of generalising on the test data is poor. Similarly, the accuracy of the training set is very good, whereas the accuracy of the validation set remains low.  
#  
#  
# We can remedy this by adding more parameters to the model in the fully connected layers, such as via Dense layers. With more model parameters, the model's ability to overfit on the  training set is reduced. To remedy the situation we included three more layers: 1x Flattening layer, one Dense layer of 512 with Relu activation and one Dropout of 0.5.  
# 

# In[10]:


batch_size = 10
image_size = (224, 224)
image_shape= (image_size[0], image_size[1], 3)
train_generator_VGG16 = train_data_augmentation(train_df, image_size, batch_size)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD

# We upload the VGG16 model without the last layer (for the output)
base_VGG16_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(image_shape))

# We create layers to merge with the VGG16 model in order to give us an output of 20 classes
head_VGG16_model = base_VGG16_model.output
head_VGG16_model = Flatten(name="flatten")(head_VGG16_model)
head_VGG16_model = Dense(512, activation="relu")(head_VGG16_model)
head_VGG16_model = Dropout(0.5)(head_VGG16_model)
head_VGG16_model = Dense(20, activation="softmax")(head_VGG16_model)


# We merge the base VGG16 model with our newly created layers
merged_VGG16_model = Model(inputs=base_VGG16_model.input, outputs=head_VGG16_model)

# We freeze the pretrained layers of the base VGG16 model
for layer in base_VGG16_model.layers:
    layer.trainable = False

# We compile the whole merged model
merged_VGG16_model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=1e-4, momentum=0.9), metrics=["accuracy"])


# Model summary 
merged_VGG16_model.summary()

# Code line below if we want to train this model

#merged_VGG16_model.fit(train_generator_VGG16,steps_per_epoch=749 // batch_size, epochs=20)   


# The current model still suffers overfitting: the training error is low, but the validation error does not reduce. This is not very satisfactory, as it points towards the model fitting well on the training data but not on the test data. Similarly, the accuracy of the training set is very well, whereas the accuracy of the validation set remains low.  
# 
# 
# We can remedy this by adding more parameters to the model, such as via Dense layers. With more model parameters, the models ability to overfit on the  training set is reduced. To remedy the situation I included three more layers: 2x Dropout of 0.2 size and a Dense layer with 200 classes and a ‘relu’ activation. 
#  
# 
# We also need to take into account that we have imbalanced classes, thus not all classes are equally represented in the dataset.

# ### 2.2.2 ResNet50V2

# After the subsequent successful architecture (AlexNet ), each new layer uses more in a deep neural network to reduce the error rate.
# If the number of layers is higher, a common problem in deep learning is associated with the gradient called vanishing or exploding. This causes the gradient to be 0 or too large. If the number of layers increases, the training and test error rate also increases. It becomes difficult to train bigger models, the accuracy is saturated and then degrades. 
# 
# In order to solve the problem of the vanishing/exploding gradient, Resnet introduced the concept called Residual Network using socalled skip connections. 
# The skip connection skips training from a few layers and connects directly to the output. 
# The approach behind this network is instead of layers learn the underlying mapping, the network is allowed to fit the residual mapping.
# 
# In doing so, the training of very dee neural networks can be done, without the problems caused by vanishing/exploding gradient.  
# The authors of the paper experimented on 100-1000 layers on CIFAR-10 dataset. The dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
# The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. 
# 
# The network architecture was inspired by VGG neural networks (VGG-16, VGG-19), with the convolutional networks having 3×3 filters. ResNets do have fewer filters and lower complexity, the ResNet50 has 50 layers. The inputs of ResNet50V2 are of fixed size of 224x224.
# The used network is Resnet50V2 which is  an improvement on the ResNet50. 

# In[11]:


batch_size = 64
image_size = (224, 224)
image_shape= (image_size[0], image_size[1], 3)
train_generator_ResNet50V2 = train_data_augmentation(train_df, image_size, batch_size)

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers


# We upload the ResNet50V2 model without the last layer (for the output)
base_ResNet50V2_model = ResNet50V2(weights="imagenet", include_top = False,input_shape=image_shape)

# We create layers to merge with the ResNet50V2 model in order to give us an output of 20 classes
head_ResNet50V2_model = base_ResNet50V2_model.layers[-1].output
head_ResNet50V2_model = Flatten(name="flatten")(head_ResNet50V2_model)
head_ResNet50V2_model = Model(inputs=base_ResNet50V2_model.input, outputs=head_ResNet50V2_model)

# We freeze the pretrained layers of the base ResNet50V2 model
for layer in head_ResNet50V2_model.layers:
    layer.trainable = False

# We merge the base ResNet50V2 model with our newly created layers
merged_ResNet50V2_model = Sequential()
merged_ResNet50V2_model.add(head_ResNet50V2_model)
merged_ResNet50V2_model.add(layers.Dense(512, activation='relu', input_dim=image_shape))
merged_ResNet50V2_model.add(layers.Dropout(0.3))
merged_ResNet50V2_model.add(layers.Dense(512, activation='relu'))
merged_ResNet50V2_model.add(layers.Dropout(0.3))
merged_ResNet50V2_model.add(layers.Dense(units=20, activation='sigmoid'))

# We compile the whole merged model
merged_ResNet50V2_model.compile(loss='binary_crossentropy',optimizer="sgd",metrics='accuracy')

# Model summary 
merged_ResNet50V2_model.summary()

# Code line below if we want to train this model

#merged_ResNet50V2_model.fit(train_generator_ResNet50V2,steps_per_epoch=749 // batch_size, epochs=20)   


# Result: Several optimizers have been tried, yet Resnet50V2 did not perform well. The model failed to "learn" on the given pretrained model and was underfitting. The accuracy did not improve across a threshold and therefore the model would not learn well on the given training dataset, as well as fail to generalize on the test set. The result on the training set was at an accuracy of 0.05, thus no valid solution. 

# ### 2.2.3 MobileNetV2

# MobileNetV2 is a CNN destined for usage on mobile devices. The basis is an inverted residual structure where the residual connections are between the bottleneck layers. 
# The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. MobileNetV2 contains two types of blocks: one residual block with stride 1 and one block with stride 2 for downsizing. The first layer is a 1×1 convolution with ReLU6, followed by a depthwise convolution. 
# The third layer is another 1×1 convolution but without any non-linearity. The architecture of MobileNetV2 has the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers. 
# 
# A key aspect of both MobileNets V1 and V2 is their use of depthwise separable convolutions, which significantly reduce the number of parameters compared to networks of the same depth but with regular convolutions. 
# The networks advantage is its small size, low-latency, low-power model that can be broadly applicable for many use cases such as image classification and object detection, but runs exceptionally well on CPUs instead of costly and resource-intensive GPUs.

# In[12]:


batch_size = 64
image_size = (256, 256)
image_shape= (image_size[0], image_size[1], 3)
train_generator_MobileNetV2 = train_data_augmentation(train_df, image_size, batch_size)

from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# We upload the MobileNetV2 model without the last layer (for the output)
base_MobileNetV2_model = MobileNetV2(weights="imagenet", include_top=False)

# We create layers to merge with the MobileNetV2 model in order to give us an output of 20 classes
head_MobileNetV2_model = base_MobileNetV2_model.output
head_MobileNetV2_model = GlobalAveragePooling2D()(head_MobileNetV2_model)
head_MobileNetV2_model = Flatten()(head_MobileNetV2_model)
#head_MobileNetV2_model = Dense(32, activation="relu")(head_MobileNetV2_model)
#head_MobileNetV2_model = Dropout(0.5)(head_MobileNetV2_model)
head_MobileNetV2_model = Dense(20, activation="softmax")(head_MobileNetV2_model)

# We merge the base MobileNetV2 model with our newly created layers
merged_MobileNetV2_model = Model(inputs=base_MobileNetV2_model.input, outputs=head_MobileNetV2_model)

# We freeze the pretrained layers of the base MobileNetV2 model
for layer in base_MobileNetV2_model.layers:
    layer.trainable = False

# We compile the whole merged model
merged_MobileNetV2_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['binary_accuracy'])

# Model summary 
merged_MobileNetV2_model.summary()

# Code line below if we want to train this model

merged_MobileNetV2_model.fit(train_generator_MobileNetV2,epochs=20) 


# Result: We can see here that the model learns somewhat. The accuracy increases yet the loss does not decrease within 10 epochs. Generalization of the MobileNet on the test set of Pascal Voc produced the best performance across the models. 

# ### 2.2.4 Fine Tuning

# One last step is fine-tuning, which is means unfreezing the entire model or parts of the model obtained and re-training it on the dataset with a small learning rate. The goal is to achieve improvements, by incrementally adapting the pretrained features to the new data.
# 
# After the initial training, we tried to further improve the performamce of our transfer learning model by unfreezing the layers of the pretrained model and perform another training session (see the code below). 
# In doing so, the weights in the lastlayers of the pretrained model (that are unfrozen) are updated given the new data. Usually, iterative unfreezing starting at the last layer is performed. 
# However, this did not lead to any increased accuracy.
# 
# As reported, MobileNetV2 produced best results on the test set so far - therefore the fine tuning for better accuracy results will be done with MobileNetV2 as a base. 

# In[13]:


len_model = len(base_MobileNetV2_model.layers)
print("Number of layers in the base model: ", len_model)


# In[14]:


from tensorflow.keras.optimizers import SGD, RMSprop

# first set: base_MobileNetV2_model.layers.trainable = True

# Fine-tune from this layer onwards (usually start at the last layer minus 1)
fine_tune_at = len_model-2

# We unfreeze the layers
for layer in base_MobileNetV2_model.layers[:fine_tune_at]:
  layer.trainable = False

merged_model = merged_MobileNetV2_model

merged_model.compile(optimizer=RMSprop(learning_rate=0.00001, momentum=0.9), loss='categorical_crossentropy', metrics=['binary_accuracy'])

# Model summary 
merged_model.summary()

#Code line below if we want to retrain this model

merged_model.fit(train_generator_MobileNetV2, epochs=5)


# Once we finished training our model, we use the following code to make our classifications on the test data. We are using the merged_MobileNetV2_model which was made by adding output layers to the frozen layers of the base of the MobileNetV2. We are not using the retrained model with the unfrozen layers because it did not improve the performance.

# In[15]:


#image_size = (256, 256)

test_img = img_resize(test_df, image_size)

threshold = 0.15

images = []
im = test_img/255
indexes = np.arange(20)
predictions = merged_MobileNetV2_model.predict(im)
for i in range(len(test_img)):
    match = False
    for ind in indexes:
        if(predictions[i, ind] > threshold):
            test_df.at[i,labels[ind]] = 1
            match = True
        else:
            test_df.at[i,labels[ind]] = 0
    if match:
        match = False
    else:
        test_df.at[i, labels[np.argmax(predictions[i])]] = 1


# # 3. Semantic segmentation
# The goal here is to implement a segmentation CNN that labels every pixel in the image as belonging to one of the 20 classes (and/or background). Use the training set to train your CNN and compete on the test set (by filling in the segmentation column in the test dataframe).

# In[16]:


get_ipython().system('mkdir -p images_to_train/x images_to_train/y')


# In[17]:



for i, j in enumerate(zip(train_df["img"].to_numpy(), train_df["seg"].to_numpy())):
    x, y = j
    Image.fromarray(x).save(f"images_to_train/x/x{i}.png","PNG")
    Image.fromarray(y).save(f"images_to_train/y/y{i}.png","PNG")


# We need to create a Dataset using Tensorflow for the images that we will use to train.

# In[18]:


#IMAGE_SIZE = 512
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_CLASSES = 21
DATA_DIR = "./images_to_train/"
NUM_TRAIN_IMAGES = 675 # change if we decide to use the data agumentation
NUM_VAL_IMAGES = 74 # change if we decide to use the data agumentation

train_images = sorted(glob(os.path.join(DATA_DIR, "x/*")))[:NUM_TRAIN_IMAGES] 
train_masks = sorted(glob(os.path.join(DATA_DIR, "y/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "x/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "y/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]


# In[19]:


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset



train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)


# In[20]:


from tensorflow.keras import layers
from tensorflow import keras


# ## DeeplabV3+
# 
# DeepLabv3+ is a model for semantic segmentation, where the goal is to assign semantic labels to every pixel in the input image. It has an encoding phase and a decoding phase, in the encoding phase it extracts the essential information of the image using CNN, while the decoding phase reconstructs the output with the information obtained in the encoder part. The decoder part was added to better segment along object boundaries, it is easy to see that DeepLabv3+ is a large model needing a great amount of processing. https://keras.io/examples/vision/deeplabv3_plus/
# 

# ### Building...
# Now we are ready to build our model!
# 
# The model that we are using here has a encoder-decoder structure, where in the encoder part the processing of contextual information happens by applying the dilated CNN at multiple scales and the decoder refine object boundaries of the segmentation.
# 
# Image from: https://www.sciencedirect.com/science/article/pii/S0167865520302750

# ![1-s2.0-S0167865520302750-gr2.jpg](https://ars.els-cdn.com/content/image/1-s2.0-S0167865520302750-gr2.jpg)
# 

# In[21]:


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


# We will use a MobileNet model pretrained on ImageNet

# In[22]:


model_input = keras.Input(shape=(224, 224, 3))
mob_net = keras.applications.MobileNet(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
mob_net.summary()


# We will also use the features (low-level) from conv_pw_11_bn to concatenate with the encoder features.

# In[23]:


def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    mob_net = keras.applications.MobileNet(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = mob_net.get_layer("conv_pw_11_bn").output
    
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = mob_net.get_layer("conv_pw_3_bn").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)


# ### Ready for the finetuning! Let’s go!
# For the optimizer we use Adam and for the loss Sparse Categorical Crossentropy
# 
# The checkpointer will save the best model
# Also was applied the learning rate reduction when the model stops learning for consecutives epochs. If the model continues not improving, there will be an early stopping.
# We also add plots to see how the model is performing.

# In[24]:


earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint("seg_model_mobilenet.h5" , verbose=1, save_best_only=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=loss,
    metrics=["accuracy"]    
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=40,
                    callbacks=[earlystopper,checkpointer,learning_rate_reduction])

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()


# Now, we are loading the saved model that was defined in the checkpointer.

# In[25]:


m = load_model("seg_model_mobilenet.h5")


# In[26]:


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        return prediction_mask


# In[27]:


get_ipython().system('mkdir -p test_images')


# In[28]:


def resize_img(imgs, width, height, interpolation=cv2.INTER_LINEAR):
    resized_imgs = []
    for mask in imgs:
        im_resize = cv2.resize(mask, 
                               (width, height), 
                               interpolation=interpolation)
        resized_imgs.append(im_resize)
    return resized_imgs


# Now,we are applying the segmentation in the test images and saving the results.

# In[29]:


final_images_seg = []
for i, j in enumerate(test_df["img"].to_numpy()):
    shape = j.shape[:2][::-1]
    im_name = f"test_images/x{i}.png"
    Image.fromarray(j).save(im_name,"PNG")
    img_infer = infer(m, read_image(im_name))
    final_images_seg.append(resize_img([img_infer], *shape, interpolation=cv2.INTER_NEAREST)[0])


# In[30]:


test_df["seg"] = final_images_seg


# In[31]:


test_df.head()


# ## Submit to competition
# 

# We have encountered some issues submitting the csv file to the competition after saving the notebook. After checking Kaggle forums and trying a few ways to fix this issue, we found out that Kaggle cannot find the submission file if the notebook has more than 500 output files. The best workaround is to produce fewer output files on the notebook version that will be used to make a submission.
# 
# Our output files are just training images used in the segmentation section and a saved segmentation model. The following few lines delete those files so that we can be able to submit the output csv file.

# In[32]:


import shutil
import os    

shutil.rmtree('images_to_train')   
shutil.rmtree('test_images') 
os.remove('seg_model_mobilenet.h5')


# The following line saves our predictions as a csv file to be submitted for the competition

# In[33]:


generate_submission(test_df)


# # 4. Adversarial attack
# 
# For this part, your goal is to fool your classification and/or segmentation CNN, using an *adversarial attack*. More specifically, the goal is build a CNN to perturb test images in a way that (i) they look unperturbed to humans; but (ii) the CNN classifies/segments these images in line with the perturbations.

# We have taken the test data image from the “Cat” class, which has been correctly classified by the model with a high confidence interval. Then, noise perturbation of different values is added to each pixel based on its contribution to the loss value. Loss calculation is done while model parameters are constant.
# We use the infinity norm perturbation, which takes the largest absolute value of an element that doesn’t exceed precision constraint epsilon. Simply, it restricts the maximum bound on the change in activation function of the neural network model.
# 
# N = E.sign(w), where N is the perturbation, w is the weights and E is the precision constant.
# 
# We produce multiple images with different precision constant values. Based on experimentation, we have observed that original images that have already been classified with high accuracy require higher noise perturbations to fool the network (they are more robust).
# 
# As the perturbation is increased, the image becomes less recognizable to the human eye.
# 
# These attacks are realistic as noise perturbation occurs constantly in a real world setting, due to sensor and camera calibrations and sensitivity, and image resolution quality. Furthermore, a real may always be different from the images used to train the models. This makes the attack very realistic.
# 
# The attack conditions to be met are the same as we mimicked in the code. Another example that also produces some “noise” in the image is when a picture is taken from an original object, printed and then photographed again. We usually distinguish some “blur”/”noise” due to the losses in taking the image, printing and retaking an image and this could be a similar approach to fooling a system. 
# 
# These attacks can also be instigated by adversaries for malicious goals. There are examples where the Google AI [6], a state of the art system was fooled. There is also other proof that shows how stickers and different angles impede correct classification of traffic signs for example [7]. As a consequence, each image recognition system needs to be handled with caution. 
# 

# In[34]:


import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False


# In[35]:


pretrained_model = merged_MobileNetV2_model
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


# In[36]:


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

def get_imagenet_label(probs):
    testNP = probs
    zeros = np.zeros(980)
    test_image_probs = np.append(testNP,zeros)
    yy = np.expand_dims(test_image_probs,0)

    return decode_predictions(yy, top=1)[0][0]#, resultingProb]


# In[37]:


advIdx = 120
originalImage = test_df["img"][advIdx]
image = preprocess(originalImage)


# In[38]:


image_probs = pretrained_model.predict(image)


# In[39]:


plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
labelPred = np.argmax(predictions[advIdx])
labelTitle = test_df.columns[labelPred]
plt.title('{} : {:.2f}% Confidence'.format(labelTitle, class_confidence*100))
plt.show()


# In[40]:


loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


# In[41]:


label = tf.one_hot(advIdx, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversarial_pattern(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]


# In[42]:


def display_images(image, description, labelTitle_Final):
    _, label, confidence, = get_imagenet_label(pretrained_model.predict(image))
    plt.figure()
    plt.imshow(image[0]*0.5+0.5)
    #plt.title('{} \n {} : {:.2f}% Confidence'.format(description,label, confidence*100))
    predictedLabel = labelTitle_Final
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description,predictedLabel, confidence*100))
    plt.show()


# In[43]:


#epsilons = [0, 0.01, 0.1, 0.55]
epsilons = [0, 0.11, 0.17, 0.28]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    newLabelProb = pretrained_model.predict(adv_x)
    labelPred = np.argmax(newLabelProb)
    labelTitle_Final = test_df.columns[labelPred]
    display_images(adv_x, descriptions[i],labelTitle_Final)


# # 5. Discussion
# 
# 
# #### Classification
# Classification of the image data has been done with a self-built CNN (with AlexNet Architecture), as well as transfer learning with the pretrained models of VGG16, ResNet50V2 and MobileNetV2. Among the four models used, the self-built CNN only improves accuracy very slowly. Resnet50V2 was underfitting and VGG16 suffers from overfitting - both models did not provide satisfactory results.  MobilenetV2 provided the best results in terms of accuracy and loss and generalizes best on the test data. 
# 
# This diverging behaviour is either a result of data preprocessing and augmentation or also the similarity of the base images, the different models have been pre-trained on. However, with more fitting of the pretrained models, 
# Finally we picked MobileNetV2 as the final model. Fine tuning was attempted with unfreezing the last layer of the pretrained MobilenedV2 sequentially. Thus we first unfroze the last layer, then the second last, etc. and tested for improvements in the accuracy score. 
# Whereas we looked at different accuracy measures such as hamming loss, accuracy, binary accuracy and categorical accuracy. 
# 
# #### Data augmentation
# Data augmentation has been applied to expand the size of a training dataset by creating alternative versions of each image in the dataset. These alternative versions of the image are created by applying different edits to the images (e.g. zoom, brighteness, flipping... etc) does not only expand the training data but also makes the model more robust to variations in the images that it will have to classify in the test data.
# 
# 
# 
# #### Which semantic segmentation technique and why? 
# DeepLabV3+, several studies [1][2][3] have used this segmentation model with different versions of the PASCAL dataset. A comparisons between models were found[4], where we can see the great performance of DeepLabV3+ using the Pascal 2012.
# 
# 
# #### Determination of threshold value proved to be difficult
# The definition of what could be the optimal threshold was taken into account and prove to be difficult to be established:
#     Some images do not have something to classify
#     Some images has just one thing to be classified while others has several classes to be predicted
#     Another issue was that sometimes the probability to have something in the image was really low and still a correct prediction, while others images have several high probabilities.
#     Several thresholds functions were tested, the better predictions came from argmax
# 
# #### Could have done a second segmentation which is promising to produce better results?
#    Yes, we could have done a second segmentation. In the late stage we found a paper [5] that used a combination of two segmentations and got a better score using DeepLabV3+ in PASCAL VOC 2012. 
# The authors got the  class index score map using DeepLabV3+ and applied the superpixels in the input image using quick shift. They explain that it is hard for DeepLabV3+ produce a semantic segmentation with accurate boundaries since it is heavy to train and then the DCNN should adopt grouping and convolution to have parameters reduction and also the generated cascade features blur the boundaries. The result was promising in most cases, but still has some difficulties when the boundaries are similar with the background.
# 
# #### Reflections on Adversarial Attack
# In the case of our model, it is especially susceptible to noise perturbations for images that are not predicted with a very high confidence score. Some of the solutions to increase robustness include:
# 
# Hiding the model gradients from adversaries, although an alternative black-box attack can overcome this defense.
# Distilling knowledge from large networks to smaller models, which contains the scope of attack.
# Making the image features less rich, so that the predicted probabilities are encoded with fewer values. This may reduce model accuracy performance.
# Using an additional model prior to testing, which filters between regular and adversarial examples.
# 
# We would like to specifically attempt the defense (3) by using PCA techniques to train on a reduced feature set data, instead of directly using the raw images and (4) use a model of continual learning (LSTM-RNN) that evaluates how the gradients change between images domains that would be otherwise similar.
# 
# Additionally, since our models appear easily fooled for images that include less rich features, such as edges and variety of colours, it would be interesting to tune parameters individually for such features.
# 
# 
# 
# 
# 
# [1] Zeng, Haibo, Siqi Peng, and Dongxiang Li. "Deeplabv3+ semantic segmentation model based on feature cross attention mechanism." Journal of Physics: Conference Series. Vol. 1678. No. 1. IOP Publishing, 2020.
# 
# [2] GitHub. 2022. GitHub - VainF/DeepLabV3Plus-Pytorch: DeepLabv3 and DeepLabv3+ with pretrained weights for Pascal VOC & Cityscapes. [online] Available at: <https://github.com/VainF/DeepLabV3Plus-Pytorch>.
# 
# [3] Papers.nips.cc. 2022. [online] Available at: <https://papers.nips.cc/paper/2019/file/a67c8c9a961b4182688768dd9ba015fe-AuthorFeedback.pdf>.
# 
# [4] Paperswithcode.com. 2022. Papers with Code - PASCAL VOC 2012 test Benchmark (Semantic Segmentation). [online] Available at: <https://paperswithcode.com/sota/semantic-segmentation-on-pascal-voc-2012?p=wider-or-deeper-revisiting-the-resnet-model> [Accessed 24 May 2022].
# 
# [5] Zhang, Sanxing & Ma, Zhenhuan & Zhang, Gang & Lei, Tao & Zhang, Rui & Cui, Yi. (2020). Semantic Image Segmentation with Deep Convolutional Neural Networks and Quick Shift. Symmetry. 12. 427. 10.3390/sym12030427. 
# 
# 
# [6] K. Eykholt et al., "Robust Physical-World Attacks on Deep Learning Visual Classification," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 1625-1634, doi: 10.1109/CVPR.2018.00175.
# 
# [7] A. Athalye et al., ”Synthesizing Robust Adversarial Examples”, Proceedings of the 35th International Conference on Machine Learning, {ICML} 2018,doi: 10.48550/arxiv.1707.07397
