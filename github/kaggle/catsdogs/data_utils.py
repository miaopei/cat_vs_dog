import numpy as np
import platform, os, random,bcolz
from PIL import Image

def one_hot_encoding(filename):
    label = filename[:filename.index('.')]
    if label == 'cat':
        return 1
    else:
        return 0

def save_array(data_folder, fname, arr):
    fname = os.path.join(data_folder, fname)
    print("Saving to {0} ...".format(fname))
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(data_folder, fname):
    fname = os.path.join(data_folder, fname)
    print("Loading from {0} ...".format(fname))
    return bcolz.open(fname)[:]

def load_data(data_folder, num_training_ratio=0.8):
    '''
    load all of cats/dogs, defer the acutal image loading
    '''
    training_folder = os.path.join(data_folder, "train")
    testing_folder = os.path.join(data_folder, "test")

    X = []
    Y = []
    for f in os.listdir(training_folder):
        # img = Image.open(os.path.join(training_folder, f)).resize(size=resize, resample=Image.ANTIALIAS)
        X.append(os.path.join(training_folder, f))
        Y.append(one_hot_encoding(f))
    X = np.array(X)
    Y = np.array(Y)

    data_size = X.shape[0]

    indicies = np.arange(data_size)
    np.random.shuffle(indicies)
    X = X[indicies]
    Y = Y[indicies]

    X_te = []
    Y_te = []
    for f in os.listdir(testing_folder):
        # img = Image.open(os.path.join(testing_folder, f)).resize(size=resize, resample=Image.ANTIALIAS)
        # X_te.append(np.asarray(img, dtype='float64'))
        X_te.append(os.path.join(testing_folder, f))
        Y_te.append(os.path.join(testing_folder, f))
    X_te = np.array(X_te)
    Y_te = np.array(Y_te)
    
    num_training = int(num_training_ratio * data_size)
    mask = range(num_training)

    X_tr = X[mask]
    Y_tr = Y[mask]

    mask = range(num_training, data_size)
    X_val = X[mask]
    Y_val = Y[mask]

    return X_tr, Y_tr, X_val, Y_val, X_te, Y_te

def load_image(X, resize=(32,32), normalize=False):
    '''
    Layze image loading
    '''
    data = [np.asarray(Image.open(f).resize(size=resize, resample=Image.ANTIALIAS), 'float64') for f in X]
    data = np.array(data)

    # Nomalize data
    if normalize:
        mean_image = np.mean(data, axis=0)
        std_imamge = np.std(data, axis=0)
        return (data - mean_image) / std_imamge
    return data

def get_minibatches(data, labels, minibatch_size, shuffle=True, resize=(32,32)):
    '''
    Genreate the mini batches
    '''
    data_size = data.shape[0]
    indicies = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indicies)
    for start_idx in range(0, data_size, minibatch_size):
        idx = indicies[start_idx : start_idx + minibatch_size]
        yield load_image(data[idx], resize), labels[idx]

if __name__ == '__main__':
    '''
    For testing purpose
    '''
    X_tr, Y_tr, X_val, Y_val, X_te, _ = load_data('datasets')

    print(X_tr.shape)
    print(Y_tr.shape)
    print(X_val.shape)
    print(Y_val.shape)

    for i, (x, y) in enumerate(get_minibatches(X_val, Y_val, 16, True)):
        print(x, y)
        break