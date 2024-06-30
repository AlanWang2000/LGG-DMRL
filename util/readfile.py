from scipy.io import loadmat
import numpy as np


def readfile(filename):
    path = './datasets/'+filename+'.mat'
    print(path)
    X, Y, V, N, clusters = None, None, None, None, None
    if filename == 'handwritten':
        dataset = loadmat(path)
        X = {0: dataset['X'][0][0].T,
            1: dataset['X'][0][1].T}
        Y = dataset['gt']
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 2

    else:
        print('cannot find this \'{}\' file!'.format(path))

    label_max = max(np.unique(Y))
    clusters = len(np.unique(Y))
    if int(label_max) == clusters:
        Y = Y-1

    out = 'file: {}  ==>  V: {}(view {} ~ {}), N: {}, clusters: {},  '.format(path, V, 0, V-1, N, clusters)
    for i in range(V):
        out += 'X[{}]: {}   '.format(i, X[i].shape)
    out += ' Y: {}'.format(Y.shape)
    print(out)

    return X, Y, V, N, clusters