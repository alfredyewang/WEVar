

"""Example of generating correlated normally distributed random samples."""
from os import listdir
from os.path import isfile, join
import numpy as np

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
np.random.seed(123)
from scipy.stats import gaussian_kde
import cloudpickle
import sys
def kde_scipy(x,bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde


def CV(m, file):

    pdfs1=[]
    pdfs2=[]

    for k in range(9):
        print(k)
        with open('./{}_pdf/{}/{}_training_1.cp.pkl'.format(file,m, k), 'rb') as f:
            kde1 = cloudpickle.load(f)
        with open('./{}_pdf/{}/{}_training_2.cp.pkl'.format(file,m,k), 'rb') as f:
            kde2 = cloudpickle.load(f)
        pdfs1.append(kde1)
        pdfs2.append(kde2)


    print(file)
    label = np.load('./data_npy/{}_label.npy'.format(file))
    score = np.load('./data_npy/{}_score.npy'.format(file))
    print(score.shape)
    new = []
    new2 = []
    print(score.shape)
    # K = [0, 1, 2, 3, 5, 7, 8, 9, 11]
    for i in range(score.shape[1]):
        print(i)
        pdf1 = pdfs1[i].evaluate(score[:, i])
        pdf2 = pdfs2[i].evaluate(score[:, i])
        new.append(pdf1)
        new2.append(pdf2)

    new = np.array(new).T
    new2 = np.array(new2).T
    # np.save('./HGMD_pdf/{}/{}_1.npy'.format(m,file), new)
    # np.save('./HGMD_pdf/{}/{}_2.npy'.format(m,file), new2)
    np.save('./{}_pdf/{}/{}_1.npy'.format(file,m,file), new)
    np.save('./{}_pdf/{}/{}_2.npy'.format(file,m,file), new2)


if __name__ == '__main__':
    data  = 'Allele_imbanlance'
    # data = 'GWAS'
    # data = 'eQTL'
    file = data.strip()
    b = 0.1
    CV(b, file)

