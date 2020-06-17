import numpy as np
from scipy.stats import gaussian_kde
import cloudpickle


def kde_scipy(x,bandwidth=0.2, **kwargs):
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde


def transform(score, methods, m=0.1):

    pdfs1=[]
    pdfs2=[]

    for k in range(9):
        with open('pdf/{}/{}/{}_training_1.cp.pkl'.format(methods,m, k), 'rb') as f:
            kde1 = cloudpickle.load(f)
        with open('pdf/{}/{}/{}_training_2.cp.pkl'.format(methods,m,k), 'rb') as f:
            kde2 = cloudpickle.load(f)
        pdfs1.append(kde1)
        pdfs2.append(kde2)
    new = []
    new2 = []
    for i in range(score.shape[1]):
        pdf1 = pdfs1[i].evaluate(score[:, i])
        pdf2 = pdfs2[i].evaluate(score[:, i])
        new.append(pdf1)
        new2.append(pdf2)

    X1 = np.array(new).T
    X2 = np.array(new2).T

    x = X2 / X1

    x = np.ma.log(x)
    x = x.filled(0)
    return x