import tensorflow as tf
import logging,math,os
import numpy as np
from utils import Progbar
from data_utils import load_data, get_minibatches
from vgg16 import vgg16

class Cats_Model(vgg16.Vgg16):
    def __init__(self):
        self.inputs_placeholder = None
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

    def load_weights(self, weights_file, sess):
        super(Cats_Model, self).load_weights(weights_file, sess)

    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 224,224, 3), name='X')
        self.labels_placeholder = tf.placeholder(dtype=tf.int64, shape=(None), name='y')
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.training_placeholder = tf.placeholder(tf.bool)

    def create_feed_dict(self, inputs_batch, labels_batch=None, is_training=True):
        feed_dict = {
            self.inputs_placeholder : inputs_batch,
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
        
        y_out = super(Cats_Model, self).add_prediction_op()

        y_out = tf.contrib.layers.batch_norm(
            inputs=y_out,
            center=True,
            activation_fn=tf.nn.relu,
            is_training=is_training
        )

        fc9_W = tf.get_variable("fc9_W", shape=(1000, 2), initializer=tf.contrib.layers.xavier_initializer())
        fc9_b = tf.get_variable("fc9_b", shape=(2), initializer=tf.zeros_initializer())
        self.trainable_varilables += [fc9_W, fc9_b]
        return tf.matmul(y_out, fc9_W) + fc9_b
        
    def add_loss_op(self, y_out):
        y = self.labels_placeholder

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(y, 2),
            logits=y_out
        )
        loss = tf.reduce_mean(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return loss, correct_prediction, accuracy
    
    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        # train_op = optimizer.minimize(loss, var_list=self.trainable_varilables)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, var_list=self.trainable_varilables)

        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, is_training=True, dropout=1):
        feed = self.create_feed_dict(inputs_batch, labels_batch, is_training)
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

        resize = (224, 224)

        for i, (train_x, train_y) in enumerate(get_minibatches(X_tr, Y_tr, batch_size,True,resize)):     
            loss, corr = self.train_on_batch(sess, train_x, train_y, True, dropout)
            prog.update(i + 1, [('train_loss', loss), ('train_acc', np.sum(corr) / train_x.shape[0])])

        prog = Progbar(target=int(math.ceil(X_val.shape[0] / batch_size)))

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

    def predict(self, sess, test_set, batch_size):
        '''
        The step to predict the image
        '''
        X_te, Y_te = test_set

        prog = Progbar(target=int(math.ceil(X_te.shape[0] / batch_size)))
        
        probs = []
        for i, (te_x, _) in enumerate(get_minibatches(X_te, Y_te, batch_size, False, resize=(224,224))):
            feed = self.create_feed_dict(te_x, labels_batch=None, is_training=False)
            probs.append(sess.run(tf.nn.softmax(self.pred), feed))
            prog.update(i + 1, None)
        probs = np.concatenate(probs)
        preds = np.argmax(probs, axis=1)
        return preds, probs

def main(debug=True):
    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = load_data('datasets')

    tf.reset_default_graph()

    model = Cats_Model()

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(tf.global_variables_initializer())
            model.load_weights(os.path.join("vgg16", "vgg16_weights.npz"), sess)
            # model.fit(
            #     sess,
            #     3,
            #     8,
            #     (X_tr, Y_tr),
            #     (X_val, Y_val),
            #     1.0
            # )
            preds, probs = model.predict(sess, (X_te, Y_te), 16)
            for pred, prob in zip(preds, probs):
                print(pred, prob[pred])
    print("")

if __name__=='__main__':
    main()