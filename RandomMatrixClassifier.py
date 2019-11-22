import numpy as np


def mp_decompose(X):
    covar = X.T @ X / X.shape[0]
    eigvals, eigvects = np.linalg.eig(covar)
    MPthresh = (1 + np.sqrt(X.shape[1] / X.shape[0]))**2
    MPsubspace = eigvects[:, eigvals > MPthresh]
    return MPsubspace


def subspace_distance(X, ss):
    X_proj = X @ ss @ ss.T
    dist = np.linalg.norm(X - X_proj, axis=1)
    return dist


class RandomMatrixClassifier:
    def __init__(self):
        self.ss_plus = 0
        self.ss_minus = 0

    def fit(self, X, y):
        # Check for non-binary target
        if np.any(np.logical_and(y != 0, y != 1)):
            raise ValueError('Target variable must be 0 or 1')
        # Split inputs
        X_plus = X[y == 1, :]
        X_minus = X[y == 0, :]
        # Decomposition: positive and negative subspaces
        self.ss_plus = mp_decompose(X_plus)
        self.ss_minus = mp_decompose(X_minus)
        print('''Found linear subspaces:  
            For positive set: dimension {0}
            For negative set: dimension {1}'''.format(
            self.ss_plus.shape[1], self.ss_minus.shape[1]))

    def predict(self, X, epsilon=0.1):
        print('Predictions using a threshold of {}'.format(epsilon))
        # Project into subspace
        dist_plus = subspace_distance(X, self.ss_plus)
        dist_minus = subspace_distance(X, self.ss_minus)
        y = 1*(dist_plus < dist_minus + epsilon)
        return y
