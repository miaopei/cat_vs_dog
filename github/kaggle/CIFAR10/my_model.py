import tensorflow as tf
import logging,math
import numpy as np
from utils import Progbar
from vgg16 import vgg16
from data_utils import get_CIFAR10_data

class CIFAR_Model(vgg16.Vgg16):
    def __init__(self):
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.training_placeholder = None
        self.trainable_varilables = []
        self.build()

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss, self.correct_prediction, self.accuracy = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='X')
        self.labels_placeholder = tf.placeholder(dtype=tf.int64, shape=(None), name='y')
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.training_placeholder = tf.placeholder(tf.bool)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1, is_training=True):
        feed_dict = {
            self.input_placeholder : inputs_batch,
            self.training_placeholder : is_training,
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        '''
        Setup the VGG model here
        ''' 
        is_training = self.training_placeholder
        
        y_out = super(CIFAR_Model, self).add_prediction_op()

        y_out = tf.contrib.layers.batch_norm(
            inputs=y_out,
            center=True,
            activation_fn=tf.nn.relu,
            is_training=is_training
        )

        fc9_W = tf.get_variable("fc9_W", shape=(1000, 10), initializer=tf.contrib.layers.xavier_initializer())
        fc9_b = tf.get_variable("fc9_b", shape=(10), initializer=tf.zeros_initializer())
        self.trainable_varilables += [fc9_W, fc9_b]
        return tf.matmul(y_out, fc9_W) + fc9_b
        
        return y_out

    def add_loss_op(self, y_out):
        y = self.labels_placeholder

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(y, 10),
            logits=y_out
        )
        loss = tf.reduce_mean(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return loss, correct_prediction, accuracy
    
    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, is_training=True, dropout=1):
        feed = self.create_feed_dict(inputs_batch, labels_batch, dropout, is_training)
        variables = [self.loss, self.correct_prediction, self.train_op]
        if not is_training:
            variables[-1] = self.accuracy
        loss, corr, _ = sess.run(
            variables, feed_dict=feed
        )
        return loss, corr

    def run_epoch(self, sess, batch_size, training_set, validation_set, dropout):
        X_tr, Y_tr = training_set
        X_val, Y_val = validation_set

        prog = Progbar(target=int(math.ceil(X_tr.shape[0] / batch_size)))

        for i, (train_x, train_y) in enumerate(get_minibatches(X_tr, Y_tr, batch_size)):     
            loss, corr = self.train_on_batch(sess, train_x, train_y, True, dropout)
            prog.update(i + 1, [('train loss', loss), ('train_acc', np.sum(corr) / train_x.shape[0])])

        val_loss, val_corr = 0, 0
        for i, (val_x, val_y) in enumerate(get_minibatches(X_val, Y_val, batch_size, False, resize)):
            loss, corr = self.train_on_batch(sess, val_x, val_y, False)
            val_loss += loss
            val_corr += np.sum(corr)
            prog.update(i + 1, [('val_loss', loss), ('val_acc', np.sum(corr) / val_x.shape[0])])
        print("Validation loss = {0:.3g} and accuracy = {1:.3g}".format(val_loss / X_val.shape[0], val_corr / X_val.shape[0]))
        
    def fit(self, sess, epoches, batch_size, training_set, validation_set, dropout):
        for epoch in range(epoches):
            print("\nEpoch {:} out of {:}".format(epoch + 1, epoches))
            self.run_epoch(sess, batch_size, training_set, validation_set, dropout)

def get_minibatches(data, labels, minibatch_size, shuffle=True):
    data_size = data.shape[0]
    indicies = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indicies)
    for start_idx in range(0, data_size, minibatch_size):
        idx = indicies[start_idx : start_idx + minibatch_size]
        yield data[idx], labels[idx]

def main(debug=True):
    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = get_CIFAR10_data()

    tf.reset_default_graph()

    model = CIFAR_Model()

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(tf.global_variables_initializer())
            model.fit(
                sess,
                5,
                32,
                (X_tr, Y_tr),
                (X_val, Y_val),
                .6
            )
    print("")

if __name__=='__main__':
    main()