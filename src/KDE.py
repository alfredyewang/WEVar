import cloudpickle
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import gaussian_kde
import os


def kde_scipy(x, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde


def kde(X,Y, b, file):

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    kf.get_n_splits(X)
    idx_fold = 0
    for train_index, test_index in kf.split(X):

        X_training, Y_training = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]
        Y_training = Y_training.reshape(Y_training.shape[0], )
        zero_idx = np.where(Y_training == 0.0)
        one_idx = np.where(Y_training == 1)

        if not os.path.exists('pdf/{}/'.format(file)):
            os.mkdir('pdf/{}'.format( file))
            os.mkdir('pdf/{}/{}'.format( file, b))

        if not os.path.exists('pdf/{}/{}/{}'.format(file, b,idx_fold)):
            os.mkdir('pdf/{}/{}/{}'.format( file, b,idx_fold))

        np.save('pdf/{}/{}/{}/X_training.npy'.format( file,b, idx_fold), X_training)
        np.save('pdf/{}/{}/{}/Y_training.npy'.format( file,b, idx_fold), Y_training)
        np.save('pdf/{}/{}/{}/X_test.npy'.format( file,b, idx_fold), X_test)
        np.save('pdf/{}/{}/{}/Y_test.npy'.format(file,b, idx_fold), Y_test)
        #

        X_training_1 = []
        X_training_2 = []

        X_test_1 = []
        X_test_2 = []
        for k in range(X_training.shape[1]):

            kde1 = kde_scipy(X_training[zero_idx[0], k], bandwidth=float(b))
            kde2 = kde_scipy(X_training[one_idx[0], k], bandwidth=float(b))
            with open('./pdf/{}/{}/{}/{}_training_1.cp.pkl'.format( file,b, idx_fold,k), 'wb') as f:
                cloudpickle.dump(kde1, f)
            with open('./pdf/{}/{}/{}/{}_training_2.cp.pkl'.format( file,b, idx_fold, k), 'wb') as f:
                cloudpickle.dump(kde2, f)

            training1 = kde1.evaluate(X_training[:, k])
            training2 = kde2.evaluate(X_training[:, k])
            X_training_1.append(training1)
            X_training_2.append(training2)

            test1 = kde1.evaluate(X_test[:, k])
            test2 = kde2.evaluate(X_test[:, k])
            X_test_1.append(test1)
            X_test_2.append(test2)

        X_training_1 = np.array(X_training_1).T
        X_training_2 = np.array(X_training_2).T

        np.save('pdf/{}/{}/{}/training_1.npy'.format(file,b, idx_fold), X_training_1)
        np.save('pdf/{}/{}/{}/training_2.npy'.format(file,b, idx_fold), X_training_2)
        np.save('pdf/{}/{}/{}/training_label.npy'.format(file, b, idx_fold), Y_training)

        X_test_1 = np.array(X_test_1).T
        X_test_2 = np.array(X_test_2).T
        np.save('pdf/{}/{}/{}/test_1.npy'.format(file,b, idx_fold), X_test_1)
        np.save('pdf/{}/{}/{}/test_2.npy'.format(file,b, idx_fold), X_test_2)
        np.save('pdf/{}/{}/{}/test_label.npy'.format(file,b, idx_fold), Y_test)

        idx_fold = idx_fold + 1
