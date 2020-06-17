import numpy as np

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def WEVar(x, methood = "HGMD"):
    beta = np.load('pdf/{}/0.1/model/model.npy'.format(methood))

    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

    res = sigmoid(np.dot(x, beta))

    return res

