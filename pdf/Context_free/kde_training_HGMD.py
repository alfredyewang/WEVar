
from scipy.stats import gaussian_kde


import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
import dill

def kde_scipy(x,bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde

def ked(b,data):
    label_training = np.load('data_npy/{}_label.npy'.format(data))
    label_training= label_training.reshape(label_training.shape[0],)
    print(label_training.shape)
    score_training = np.load('data_npy/{}_score.npy'.format(data))
    zero_idx = np.where(label_training==0.0)
    # print(zero_idx)
    one_idx = np.where(label_training==1)
    print(score_training.shape)
    for k in range(score_training.shape[1]):
        print(k)
        # print(score_training[zero_idx[:], k].shape)
        kde1 = kde_scipy(score_training[zero_idx[0], k], bandwidth=float(b))
        kde2 = kde_scipy(score_training[one_idx[0], k], bandwidth=float(b))
        with open('./{}_pdf/{}/{}_training_1.cp.pkl'.format(data,b, k), 'wb') as f:
            cloudpickle.dump(kde1, f)
        with open('./{}_pdf/{}/{}_training_2.cp.pkl'.format(data,b,k), 'wb') as f:
            cloudpickle.dump(kde2, f)

if __name__ == '__main__':
    Z=[0.1]
    # data  = 'Allele_imbanlance'
    # data = 'GWAS'
    data = 'eQTL'
    for b in Z:
        ked(b, data)