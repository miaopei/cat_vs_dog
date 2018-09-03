import bcolz
import numpy as np
from keras import backend as K
from keras.utils.generic_utils import Progbar
import pickle

def save_array(arr, fname):
    print('Saving numpy array to %s' % fname)
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    print('Loading numpy array to %s' % fname)
    return bcolz.open(fname)[:]

def predict(model, generator, steps):
    prog = Progbar(steps)
    preds = []
    for i, batch in enumerate(generator):
        preds.append(model.predict_on_batch(batch))
        prog.update(i + 1)
    print("")
    return preds

def evaluate(model, generator, steps):
    prog = Progbar(steps)
    ret = {'loss':[], 'acc':[]}
    for i, (x, y) in enumerate(generator):
        loss, acc = model.test_on_batch(x, y)
        ret['loss'].append(loss)
        ret['acc'].append(acc)
        prog.update(i + 1, [('loss', loss), ('acc', acc)])
    print("")
    return ret

def dump(obj, filename):
    with open(filename, 'wb') as f:
        print("Saving to %s" % filename)
        pickle.dump(obj, f)

def unpickle(filename):
    with open(filename, 'rb') as f:
        print("Loading from %s" % filename)
        return pickle.load(f)