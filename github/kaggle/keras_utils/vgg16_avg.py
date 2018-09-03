import keras
import numpy as np
from keras.layers import Input, Conv2D, AveragePooling2D, Dense, Flatten, Dropout, Activation, Lambda
from keras.models import Model
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import backend as K

class Vgg16_avg(object):
    def __init__(self, input_shape=None, include_top=True):
        self.build(input_shape, include_top)
        self.model.summary()
        self.load_weights(include_top)
        
    def build(self, input_shape, include_top):
        '''
        Build a vgg16 model using Keras functional API
        https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
        '''
        #vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 1, 3))
        if input_shape is None:
            img_input = Input(shape=(None,None,3), name='x')
        else:
            img_input = Input(shape=input_shape, name='x')
        
        #x = Lambda(lambda x : x -  vgg_mean)(self.input)
        x = img_input
        
        x = Conv2D(64, 3, strides=1, padding='same', activation='relu', name='CONV1_1', trainable=False)(x)
        x = Conv2D(64, 3, strides=1, padding='same', activation='relu', name='CONV1_2', trainable=False)(x)
        x = AveragePooling2D(2, strides=2, padding='valid')(x) # 112
        
        x = Conv2D(128, 3, strides=1, padding='same', activation='relu', name='CONV2_1', trainable=False)(x)
        x = Conv2D(128, 3, strides=1, padding='same', activation='relu', name='CONV2_2', trainable=False)(x)
        x = AveragePooling2D(2, strides=2, padding='valid')(x) # 56
        
        x = Conv2D(256, 3, strides=1, padding='same', activation='relu', name='CONV3_1', trainable=False)(x)
        x = Conv2D(256, 3, strides=1, padding='same', activation='relu', name='CONV3_2', trainable=False)(x)
        x = Conv2D(256, 3, strides=1, padding='same', activation='relu', name='CONV3_3', trainable=False)(x)
        x = AveragePooling2D(2, strides=2, padding='valid')(x) # 28
        
        x = Conv2D(512, 3, strides=1, padding='same', activation='relu', name='CONV4_1', trainable=False)(x)
        x = Conv2D(512, 3, strides=1, padding='same', activation='relu', name='CONV4_2', trainable=False)(x)
        x = Conv2D(512, 3, strides=1, padding='same', activation='relu', name='CONV4_3', trainable=False)(x)
        x = AveragePooling2D(2, strides=2, padding='valid')(x) # 14
        
        x = Conv2D(512, 3, strides=1, padding='same', activation='relu', name='CONV5_1', trainable=False)(x)
        x = Conv2D(512, 3, strides=1, padding='same', activation='relu', name='CONV5_2', trainable=False)(x)
        x = Conv2D(512, 3, strides=1, padding='same', activation='relu', name='CONV5_3', trainable=False)(x)
        x = AveragePooling2D(2, strides=2, padding='valid')(x) # 7
        
        if include_top:
            x = Flatten()(x)
            x = Dense(4096, activation='relu', trainable=False)(x)
            x = Dense(4096, activation='relu', trainable=False)(x)
            y = Dense(1000, activation='softmax', trainable=False)(x)
        else:
            y = GlobalAveragePooling2D()(x)
        
        self.model = Model(img_input, y)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc']
        ) 

        self.input = self.model.input
        
    def load_weights(self, include_top):
        if include_top:
            filename = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        else:
            filename = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        print('Loading weights from "%s"' % (filename))
        self.model.load_weights(filename)

    def extract_features(self, verbose=False):
        '''
        https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
        Inputs
        - x : the input image (N, H, W, C)
        Output
        - features: the feature map of each layer
        '''
        layers = [layer.output for layer in self.model.layers if type(layer) is Conv2D]
        return layers

