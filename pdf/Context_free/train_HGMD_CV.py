from __future__ import division
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
# import real_data
np.random.seed(1)
from sklearn import metrics
def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def CV(X_tr,Y_tr,X_te,Y_te, lambda_vals, SAVE, data):

    X_tr= np.concatenate((X_tr, np.ones((X_tr.shape[0],1))),axis=1)
    n=X_tr.shape[1]
    m=X_tr.shape[0]
    Y_tr= Y_tr.reshape(m, 1)

    X_te = np.concatenate((X_te, np.ones((X_te.shape[0], 1))), axis=1)
    Y_te = Y_te.reshape(X_te.shape[0], 1)

    test = np.concatenate((np.ones(n-1), 0.0), axis=None).reshape(1,n)

    beta = cp.Variable((n,1))
    res = np.array([1])
    constraints = [beta >= 0,
                   test@beta==res]
    lambd = cp.Parameter(nonneg=True)
    log_likelihood = cp.sum(
        cp.reshape(cp.multiply(Y_tr, X_tr @ beta), (m,)) -
        cp.log_sum_exp(cp.hstack([np.zeros((m,1)), X_tr @ beta]), axis=1)
        -lambd * cp.norm(beta, 2)
    )
    problem = cp.Problem(cp.Maximize(log_likelihood),constraints)

    beta_vals = []
    lambd.value = lambda_vals
    problem.solve()
    beta_vals.append(beta.value)
    # print("status:", problem.status)
    # print("optimal value", problem.value)
    # print("optimal var", beta.value)
    if SAVE:
        np.save('./{}_pdf/{}/model/model.npy'.format(data,b), beta.value)

    res2 = sigmoid(np.dot(X_te, beta.value))
    auc = metrics.roc_auc_score(Y_te, res2)
    return auc, beta.value


if __name__ == '__main__':


    # data  = 'Allele_imbanlance'
    # data = 'GWAS'
    data = 'eQTL'
    b = 0.1
    X1 = np.load('./{}_pdf/{}/{}_1.npy'.format(data,b,data))
    X2 = np.load('./{}_pdf/{}/{}_2.npy'.format(data,b,data))
    # X1 = np.load('./HGMD_pdf/{}/HGMD_1.npy'.format(b))
    # X2 = np.load('./HGMD_pdf/{}/HGMD_2.npy'.format(b))
    score = X2 / X1
    print(score.shape)


    score = np.ma.log(score)
    score = score.filled(0)
    label = np.load('./data_npy/{}_label.npy'.format(data))
    # label = np.load('./data_npy/HGMD_label.npy')

    label = label.reshape(label.shape[0], 1)
    print(np.max(score), np.min(score))


    trials = 10
    lambda_vals = np.linspace(0.01, 0.1, trials)
    # print(lambda_vals)
    # exit(222)
    auc_mean=[]
    for i in range(trials):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits(score)
        lambda_v = lambda_vals[i]
        AUC=[]
        BETA =[]
        print(i)
        print(lambda_v)
        for train_index, test_index in kf.split(score):
            train_x, train_y = score[train_index], label[train_index]
            test_x, test_y = score[test_index], label[test_index]
            auc, beta= CV( train_x, train_y,test_x, test_y, lambda_v, False, data)
            AUC.append(auc)
            BETA.append(beta)
            print(beta)
            # print(auc)
        print('Mean:{}'.format(np.mean(np.array([AUC]))))
        auc_mean.append(np.mean(np.array([AUC])))
    idx = np.argmax(np.array(auc_mean))
    print(idx)
    CV(score, label, score, label, lambda_vals[idx], True, data)

