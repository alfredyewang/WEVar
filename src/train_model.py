import KDE
import cvxpy as cp
import numpy as np
np.random.seed(1)
from decimal import Decimal
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import seaborn as sns
np.random.seed(123)
sns.set_style("ticks")
sns.despine()
cur = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804), (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), (0.5764705882352941, 0.47058823529411764, 0.3764705882352941), (0.8549019607843137, 0.5450980392156862, 0.7647058823529411), (0.5490196078431373, 0.5490196078431373, 0.5490196078431373), (0.8, 0.7254901960784313, 0.4549019607843137) , (1.0, 0.8509803921568627, 0.1843137254901961), (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]

order = [1,0,2,3,4,7,6,5,8,9,10]


cur = [cur[i] for i in order]
def evaluation( test_hat_y, test_y):
    auc = metrics.roc_auc_score(test_y, test_hat_y)
    auc = Decimal(auc).quantize(Decimal('0.0000'))
    print(auc)
    return auc


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def CV(kfold, name, X_tr,Y_tr,X_te,Y_te,lambda_vals, kde_bandwidth,SAVE=True):


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

    res = sigmoid(np.dot(X_tr, beta.value))
    res2 = sigmoid(np.dot(X_te, beta.value))
    auc = metrics.roc_auc_score(Y_te, res2)

    if SAVE:
        if not os.path.exists('pdf/{}/{}/{}/model'.format(name, kde_bandwidth,kfold)):
            os.mkdir('pdf/{}/{}/{}/model'.format(name, kde_bandwidth,kfold))
        np.save('./pdf/{}/{}/{}/model/model.npy'.format(name,kde_bandwidth,kfold), beta.value)

    return auc


def train(X,Y, file,kde_bandwidth=0.1, num_fold= 10):

    KDE.kde(X, Y, kde_bandwidth, file)
    AUC_all =[]
    for kfold in range(num_fold):
        X_training_1=np.load('./pdf/{}/{}/{}/training_1.npy'.format(file,kde_bandwidth, kfold))
        X_training_2=np.load('./pdf/{}/{}/{}/training_2.npy'.format(file,kde_bandwidth, kfold))
        X_test_1=np.load('./pdf/{}/{}/{}/test_1.npy'.format(file,kde_bandwidth, kfold))
        X_test_2=np.load('./pdf/{}/{}/{}/test_2.npy'.format(file, kde_bandwidth,kfold))

        X_training = X_training_2 / X_training_1
        X_training = np.ma.log(X_training)
        X_training = X_training.filled(0)
        Y_training = np.load('./pdf/{}/{}/{}/training_label.npy'.format(file,kde_bandwidth,kfold))
        Y_training = Y_training.reshape(Y_training.shape[0], 1)
        X_test = X_test_2 / X_test_1
        X_test = np.ma.log(X_test)
        X_test = X_test.filled(0)
        Y_test = np.load('./pdf/{}/{}/{}/test_label.npy'.format(file,kde_bandwidth,kfold))
        Y_test = Y_test.reshape(Y_test.shape[0], 1)
        trials = 10
        lambda_vals = np.linspace(0.01, 0.1, trials)
        AUC_CV = []
        for i in range(lambda_vals.shape[0]):
            lambda_v = lambda_vals[i]
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True)
            kf.get_n_splits(X_training)
            auc_CV =[]
            for train_index, test_index in kf.split(X_training):
                x_training, y_training = X_training[train_index], Y_training[train_index]
                x_test, y_test = X_training[test_index], Y_training[test_index]
                auc_cv = CV(kfold, file,x_training, y_training, x_test, y_test ,lambda_v, SAVE=False,kde_bandwidth = kde_bandwidth)
                auc_CV.append(auc_cv)
            AUC_CV.append(np.mean(np.array(auc_CV)))

        idx = np.argmax(np.array(AUC_CV))
        auc = CV(kfold, file,X_training, Y_training, X_test, Y_test, lambda_vals[idx],SAVE=True, kde_bandwidth=kde_bandwidth)
        AUC_all.append(auc)


def test(X,Y,file,kde_bandwidth=0.1, num_fold= 10):
    fig, ax = plt.subplots(figsize=(12, 8))

    AUC = []
    AUC_PR = []
    r2 = []
    auc_wevar =[]
    auc_pr_wevar=[]
    r2_wevar =[]
    for kfold in range(num_fold):

        beta = np.load('pdf/{}/{}/{}/model/model.npy'.format(file,kde_bandwidth,kfold))
        X1= np.load('pdf/{}/{}/{}/test_1.npy'.format(file,kde_bandwidth,kfold))
        X2= np.load('pdf/{}/{}/{}/test_2.npy'.format(file,kde_bandwidth,kfold))
        x = X2 / X1
        x = np.ma.log(x)
        x = x.filled(0)
        label_wevar = np.load('pdf/{}/{}/{}/Y_test.npy'.format(file,kde_bandwidth,kfold))
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        res = sigmoid(np.dot(x, beta))
        auc = metrics.roc_auc_score(label_wevar, res)
        precision, recall, thresholds = precision_recall_curve(label_wevar, res)
        auc_pr = metrics.auc(recall, precision)
        r = np.corrcoef(label_wevar, res.reshape(res.shape[0],))
        auc_wevar.append(auc)
        auc_pr_wevar.append(auc_pr)
        r2_wevar.append(r[0,1])
    label = Y
    score0 = X
    score1 = X
    for i in range(score1.shape[1]):
        score1[:, i] = (score1[:, i] - score1[:, i].min()) / (score1[:, i].max() - score1[:, i].min())
    baseline = score1.mean(axis=1).reshape(score1.shape[0],1)
    score = np.concatenate((score0, baseline), axis=1)

    method = [
        'Eigen', 'CADD', 'DANN', 'FATHMM_MKL', 'FunSeq2', 'LINSIGHT',"GWAVA_Region",'GWAVA_TSS',"GWAVA_Unmatched", "Unweighted"]

    for k, m in enumerate(method):
        auc = metrics.roc_auc_score(label, score[:, k])
        AUC.append(auc)
        precision, recall, thresholds = precision_recall_curve(label, score[:, k])
        auc_pr = metrics.auc(recall, precision)
        AUC_PR.append(auc_pr)
        r = np.corrcoef(label.reshape(label.shape[0],),score[:,k])
        r2.append(r[0,1])
    AUC.append(np.array([auc_wevar]).mean())
    AUC_PR.append(np.array([auc_pr_wevar]).mean())
    r2.append(np.array([r2_wevar]).mean())
    method.append('WEVar')
    df_testing = pd.DataFrame({'Methods': method, 'AUROC': AUC, 'AUPR': AUC_PR, 'R2': r2})
    ax = sns.scatterplot(x="AUPR", y="AUROC", hue="Methods", size="R2",palette=cur,
                    hue_order=["CADD","Eigen", 'LINSIGHT',"DANN","GWAVA_Unmatched","GWAVA_TSS","GWAVA_Region", "FunSeq2", "FATHMM_MKL",'Unweighted','WEVar'],
                    sizes= (100,500),data=df_testing)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)
    ax.legend(handles[1:12], labels[1:12],bbox_to_anchor=(1.05, 1.0), loc='upper left',fontsize=16, frameon=False)
    ax.set_title('{}'.format(file), fontsize=24)
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.setp(ax.set_xlabel('AUPR'), fontsize=18)
    plt.setp(ax.set_ylabel('AUROC'), fontsize=18)
    sns.despine()
    plt.tight_layout()
    plt.savefig('pdf/{}/{}/{}.png'.format(file,kde_bandwidth,file))
