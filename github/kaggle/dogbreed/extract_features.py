import keras

from keras.layers import Dense
from keras.models import Model

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

import numpy as np
import os
import pickle

import sys

import h5py

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def get_classes(path):
    classes = [subdir for subdir in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, subdir))]
    class_indices = dict(zip(classes, range(len(classes))))
    return classes, class_indices

def get_image_tensors(path):
    classes, class_indices = get_classes(path)
    x = []
    for class_name in classes:
        class_dir = os.path.join(path, class_name)
        for f in os.listdir(class_dir):
            img = image.load_img(os.path.join(class_dir, f), target_size=(224,224))
            x.append(image.img_to_array(img))
    return np.array(x)

def get_labels(path):
    classes, class_indices = get_classes(path)
    y = []
    for class_name in classes:
        class_dir = os.path.join(path, class_name)
        for f in os.listdir(class_dir):
            l = np.zeros(shape=(len(classes),))
            l[class_indices[class_name]] = 1
            y.append(l)
    return np.array(y)

def train_loader(path, shuffle=True, batch_size=32, target_size=(224,224), preprocessing_function=preprocess_input):
    datagen = image.ImageDataGenerator(
        # rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocessing_function
    )

    return datagen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size, 
        shuffle=shuffle,
    )

def val_loader(path, shuffle=False, batch_size=32, target_size=(224,224), preprocessing_function=preprocess_input):
    datagen = image.ImageDataGenerator(
        # rescale=1./255,
        preprocessing_function=preprocessing_function
    )

    return datagen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size, 
        shuffle=shuffle,
    )

def predict(model, loader):
    features, labels = [], []
    num_samples = 0
    for i, (x, y) in enumerate(loader):
        if i >= len(loader):break
        num_samples += len(y)
        sys.stdout.write("%d\r" % num_samples)
        sys.stdout.flush()
        features.extend(model.predict(x))
        labels.extend(y)
    print('')
    return np.array(features), np.array(labels)

def inception_preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':

    models = {
        'inception_v3' : keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet'),
        'inception_resnet_v2' : keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet'),
        'resnet50' : keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet'),
        'vgg16': keras.applications.vgg16.VGG16(include_top=False, weights='imagenet'),
    }

    for m in models:
        model = models[m]
        print('Running model = %s' % m)
        target_size = (224, 224)
        preprocess_fn = preprocess_input
        if 'inception' in m:
            target_size = (299, 299)
            preprocess_fn = inception_preprocess_input
        loader = train_loader('datasets/train', target_size=target_size, preprocessing_function=preprocess_fn)
        train_x, train_y = predict(model, loader)

        loader = val_loader('datasets/val', target_size=target_size, preprocessing_function=preprocess_fn)
        val_x, val_y = predict(model, loader)

        loader = val_loader('datasets/test', target_size=target_size, preprocessing_function=preprocess_fn)
        test_x, _ = predict(model, loader)

        with h5py.File('datasets/' + m + '_test.h5', 'w') as f:
            f['train_features'] = train_x
            f['train_labels'] = train_y
            f['val_features'] = val_x
            f['val_labels'] = val_y
            f['test_features'] = test_x
