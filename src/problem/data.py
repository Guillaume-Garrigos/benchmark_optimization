import numpy as np
from numpy.random import multivariate_normal, randn
from scipy.linalg.special_matrices import toeplitz
from sklearn.datasets import load_svmlight_file

import logging
from src.parameters import get_config
import os

""" Data() is just a fancy dict gathering the couples (feature, label)
    needed to define our problem (see the class Problem() )
"""

##################################################
class Data():
    # basically contains the features and labels of a dataset
    def __init__(self):
        self.feature = None
        self.label = None
        self.name = ""
        self.path = ""
        self.nb_data = None
        self.dim = None
    
    def read(self, dataset_name, dataset_path_folder):
        # Given data (=features and labels) stored somewhere, loads it
        # must be stored in a file dataset_path_folder/dataset_name.txt
        self.name = dataset_name
        self.path = os.path.join(dataset_path_folder, dataset_name) + '.txt'
        X, y = get_data(self.path)
        X = X.toarray()  # convert from scipy sparse matrix to dense if necessary
        logging.info("Data Sparsity: {}".format(sparsity(X)))
        problem_type = get_config()['problem']['type']
        if problem_type == 'classification':
            # label preprocessing for some dataset whose label is not {-1, 1}
            max_v, min_v = np.max(y), np.min(y)
            idx_min = (y == min_v)
            idx_max = (y == max_v)
            y[idx_min] = -1
            y[idx_max] = 1
        elif problem_type == 'regression':
            X = X / np.max(np.abs(X), axis=0)  # scale features to [-1, 1]
        else:
            raise Exception("Unknown problem type!")
        # adding a column filling with all ones.(bias term)
        X = np.c_[X, np.ones(X.shape[0])]

        n, d = X.shape
        logging.info("Number of data points: {:d}; Number of features: {:d}".format(n, d))
        self.feature = X
        self.label = y
        self.nb_data = n
        self.dim = d
##################################################

def get_data(data_path):
    """Once datasets are downloaded, load datasets in LibSVM format."""
    data = load_svmlight_file(data_path)
    return data[0], data[1]


def sparsity(data):
    # calculate data sparsity, i.e., number of zero entries / total entries in data matrix.
    # data, (num_samples, num_features)
    n, d = data.shape
    total_entries = n * d
    zeros_entries = np.sum(data == 0)
    return zeros_entries / total_entries

# ===============================
# These two functions to generate artificial data. Not used.
# =================================

def simu_linreg(x, n, std=1., corr=0.5):
    """Simulation for the least-squares problem.

    Parameters
    ----------
    x : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix

    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The targets.
    """
    d = x.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n)
    noise = std * randn(n)
    b = A.dot(x) + noise
    return A, b


def simu_logreg(x, n, std=1., corr=0.5):
    """Simulation for the logistic regression problem.

    Parameters
    ----------
    x : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix

    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The targets.
    """
    A, b = simu_linreg(x, n, std=1., corr=corr)
    return A, np.sign(b)
# ======================================
