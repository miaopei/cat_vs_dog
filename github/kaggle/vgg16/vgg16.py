from .model import Model
import tensorflow as tf
import numpy as np
# from imagenet_classes import class_names
from scipy.misc import imread, imresize
from PIL import Image

class Vgg16(Model):
    def __init__(self):
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.build()

    def load_weights(self, weights_file, sess):
        weights = np.load(weights_file)
        keys = sorted(weights.keys())
        variables = tf.global_variables()[:len(keys)]

        for i, k in enumerate(keys):
            print(i, k, np.shape(keys), variables[i].name)
            sess.run(variables[i].assign(weights[k]))

    def add_conv3(self, inputs, num_filters, activation_fn=tf.nn.relu, scope=None):
        '''
        CONV3
        kernel_size=3
        '''
        return tf.contrib.layers.conv2d(
            inputs=inputs,
            num_outputs=num_filters,
            kernel_size=3,
            padding='SAME',
            activation_fn=activation_fn,
            scope=scope
        )
    
    def add_max_pool(self, inputs):
        return tf.contrib.layers.max_pool2d(
            inputs=inputs,
            kernel_size=2,
            stride=2,
            padding='VALID',
        )

    def add_fc(self, inputs, num_outputs, activation_fn=tf.nn.relu, scope=None):
        return tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=num_outputs,
            scope=scope,
            activation_fn=activation_fn
        )

    def add_prediction_op(self):
        X = self.inputs_placeholder # 224x224x3

        vgg_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=(1, 1, 1, 3), name='vgg_mean')

        X -= vgg_mean

        with tf.name_scope('conv1_1') as scope:
            a = self.add_conv3(X, 64, scope=scope)
        with tf.name_scope('conv1_2') as scope:
            a = self.add_conv3(a, 64, scope=scope)
        a = self.add_max_pool(a)

        with tf.name_scope('conv2_1') as scope:
            a = self.add_conv3(a, 128, scope=scope)
        with tf.name_scope('conv2_2') as scope:
            a = self.add_conv3(a, 128, scope=scope)
        a = self.add_max_pool(a)

        with tf.name_scope('conv3_1') as scope:
            a = self.add_conv3(a, 256, scope=scope)
        with tf.name_scope('conv3_2') as scope:
            a = self.add_conv3(a, 256, scope=scope)
        with tf.name_scope('conv3_3') as scope:
            a = self.add_conv3(a, 256, scope=scope)
        a = self.add_max_pool(a)

        with tf.name_scope('conv4_1') as scope:
            a = self.add_conv3(a, 512, scope=scope)
        with tf.name_scope('conv4_2') as scope:
            a = self.add_conv3(a, 512, scope=scope)
        with tf.name_scope('conv4_3') as scope:
            a = self.add_conv3(a, 512, scope=scope)
        a = self.add_max_pool(a)

        with tf.name_scope('conv5_1') as scope:
            a = self.add_conv3(a, 512, scope=scope)
        with tf.name_scope('conv5_2') as scope:
            a = self.add_conv3(a, 512, scope=scope)
        with tf.name_scope('conv5_3') as scope:
            a = self.add_conv3(a, 512, scope=scope)
        a = self.add_max_pool(a)

        a = tf.contrib.layers.flatten(a)

        with tf.name_scope('fc6') as scope:
            a = self.add_fc(a, 4096, scope=scope)
        with tf.name_scope('fc7') as scope:
            a = self.add_fc(a, 4096, scope=scope)

        with tf.name_scope('fc8') as scope:
            a = self.add_fc(a, 1000, None, scope)

        pred = a

        return pred

    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, 224,224, 3))

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {
            self.inputs_placeholder : inputs_batch,
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def predict(self, sess, inputs):
        '''
        The step to predict the image
        '''
        feed_dict = self.create_feed_dict(inputs)
        prob = sess.run(tf.nn.softmax(self.pred), feed_dict)
        
        preds = np.argmax(prob, axis=1)
        return preds

    def add_loss_op(self, pred):
        return None

    def add_training_op(self, loss):
        pass
        
        
if __name__=='__main__':
    img = Image.open('cat.1.jpg').resize((224,224))
    print(img)
    
    model = Vgg16()
    # print([x.name for x in tf.global_variables()])
    # img = imread('laska.png', mode='RGB')
    # img = imresize(img, (224,224))
    
    X = np.array([np.asarray(img), np.asarray(img)])
    # X = (X - X.mean(axis=0)) / X.std(axis=0)
    print(X.shape)
    with tf.Session() as sess:
        model.load_weights('vgg16_weights.npz', sess)
        model.predict(sess, X)