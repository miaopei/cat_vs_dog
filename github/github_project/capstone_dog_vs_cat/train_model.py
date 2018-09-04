import seaborn as sns
import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50  
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, TensorBoard
import random
import cv2


from keras.preprocessing.image import ImageDataGenerator

target_image_size = (224, 224)

train_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
        'image/img_train',
        target_size=target_image_size,  # resize
        batch_size=16,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
        'image/img_valid', 
        target_size=target_image_size,  # resize
        batch_size=16,
        class_mode='binary')



def train_func(loss_name,optimizer_name):
    base_model = ResNet50(input_tensor=Input((224, 224, 3)), weights='imagenet', include_top=False)
    for layers in base_model.layers:
        layers.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(base_model.input, x)
    model.compile(loss=loss_name, optimizer=optimizer_name, metrics=['accuracy'])
    best_model = ModelCheckpoint("resnet_best_{}_{}.h5".format(loss_name,optimizer_name), monitor='val_acc', verbose=0, save_best_only=True)
    tensor_log = "./resnet_best_{}_{}_tensor_log".format(loss_name,optimizer_name)
    return model,best_model,tensor_log


model,best_model,tensor_log = train_func("binary_crossentropy","adadelta")
model.fit_generator(
        train_generator,
        samples_per_epoch=2048,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=1024,
        callbacks=[best_model, TensorBoard(log_dir=tensor_log)])

model,best_model,tensor_log = train_func("binary_crossentropy","adam")
model.fit_generator(
        train_generator,
        samples_per_epoch=2048,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=1024,
        callbacks=[best_model, TensorBoard(log_dir=tensor_log)])

model,best_model,tensor_log = train_func("binary_crossentropy","sgd")
model.fit_generator(
        train_generator,
        samples_per_epoch=2048,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=1024,
        callbacks=[best_model, TensorBoard(log_dir=tensor_log)])









