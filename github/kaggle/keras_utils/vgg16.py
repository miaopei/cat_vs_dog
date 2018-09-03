import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam

class VGG16(object):
    def __init__(self):
        self.model = Sequential()
    
    def add_Conv(self, filters, kernel_size, strides=1, activation='relu', input_shape=None):
        if input_shape is None:
            self.model.add(Conv2D(filters, kernel_size, strides=strides, padding='same', activation=activation))
        else:
            self.model.add(Conv2D(filters, kernel_size, strides=strides, padding='same', activation=activation, input_shape=input_shape))

    def add_MaxPooling(self, pool_size=2, strides=2):
        self.model.add(MaxPooling2D(pool_size, strides=strides, padding='valid'))

    def add_FC(self, units, activation='relu'):
        self.model.add(Dense(units, activation=activation))

    def build(self, input_shape=(224, 224, 3), include_fc=True):

        vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 3))

        self.model.add(Lambda(lambda x : x - vgg_mean, input_shape=input_shape))

        self.add_Conv(64, 3)
        self.add_Conv(64, 3)
        self.add_MaxPooling()

        self.add_Conv(128, 3)
        self.add_Conv(128, 3)
        self.add_MaxPooling()

        self.add_Conv(256, 3)
        self.add_Conv(256, 3)
        self.add_Conv(256, 3)
        self.add_MaxPooling()

        self.add_Conv(512, 3)
        self.add_Conv(512, 3)
        self.add_Conv(512, 3)
        self.add_MaxPooling()

        self.add_Conv(512, 3)
        self.add_Conv(512, 3)
        self.add_Conv(512, 3)
        self.add_MaxPooling()

        if include_fc:
            self.model.add(Flatten())
            self.add_FC(4096)
            self.model.add(Dropout(.5))
            self.add_FC(4096)
            self.model.add(Dropout(.5))
            self.add_FC(1000, 'softmax')
        else:
            '''
            https://arxiv.org/pdf/1312.4400.pdf
            Section 3.2 Global Average Pooling (GAP)
            '''
            self.model.add(GlobalAveragePooling2D())
            self.model.add(Activation('softmax'))

        self.make_layers_untrainable()

    def make_layers_untrainable(self):
        for layer in self.model.layers:
            if layer.trainable:
                layer.trainable = False

    def compile(self, lr=1e-3):
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def fit(self, X, Y, batch_size=32, epochs=1, validation_split=0.0):
        return self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def fit_generator(self,generator, steps_per_epoch, epochs=1, validation_data=None, validation_steps=None):
        return self.model.fit_generator(generator, steps_per_epoch, epochs=epochs, validation_data=validation_data, validation_steps=validation_steps, max_queue_size=1)

    def predict(self, X, batch_size=32):
        return self.model.predict(X, batch_size=batch_size)

    def evaluate(self, X, Y, batch_size=32):
        return self.model.evaluate(X, Y, batch_size=batch_size)

    def load_weights(self, weights_file):
        '''
        https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5

        https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
        '''
        print('Loading weights from %s' % weights_file)
        self.model.load_weights(weights_file)

if __name__=='__main__':
    from PIL import Image
    from imagenet_classes import class_names

    img = Image.open('laska.png').convert('RGB').resize((224,224))
    x = np.asarray(img)
    x = np.array([x])

    model = VGG16()
    model.build((224,224,3))
    model.summary()
    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    probs = model.predict(x)[0]
    p = np.argmax(probs)

    print(class_names[p], probs[p])