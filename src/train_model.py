import KDE


def train(X,Y, file):
    b = 0.1
    KDE.kde(X, Y, b, file)