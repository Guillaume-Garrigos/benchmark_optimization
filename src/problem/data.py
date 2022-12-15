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
        self.config = get_config()
    
    def read(self, dataset_name, dataset_path_folder=None):
        # Obtain the data. Either loaded or generated
        self.name = dataset_name
        if dataset_name == "random":
            X, y = self.generate_data()
        else:
            self.path = os.path.join(dataset_path_folder, dataset_name) + '.txt'
            X, y = load_data(self.path)
            X = X.toarray()  # convert from scipy sparse matrix to dense if necessary
        X, y = self.preprocess_data(X, y)
        n, d = X.shape
        logging.info("Number of data points: {:d}; Number of features: {:d}".format(n, d))
        self.feature = X
        self.label = y
        self.nb_data = n
        self.dim = d
    
    def generate_data(self):
        problem_type = self.config['problem']['type']
        nb_data = 150
        nb_features = 50
        std = 0.0
        if problem_type == 'phase retrieval':
            x = np.random.randn(nb_features)
            x = x/np.linalg.norm(x)
            A = np.random.randn(nb_data, nb_features)
            noise = std* np.random.randn(nb_data)
            b = (A @ x)**2 + noise
        return A, b

    def preprocess_data(self, X, y):
        logging.info("Data Sparsity: {}".format(sparsity(X)))
        problem_type = self.config['problem']['type']
        if problem_type == 'classification':
            # label preprocessing for some dataset whose label is not {-1, 1}
            max_v, min_v = np.max(y), np.min(y)
            idx_min = (y == min_v)
            idx_max = (y == max_v)
            y[idx_min] = -1
            y[idx_max] = 1
        elif problem_type in ['regression'] :
            X = X / np.max(np.abs(X), axis=0)  # scale features to [-1, 1]
        elif problem_type in ['phase retrieval'] :
            pass
        else:
            raise Exception("Unknown problem type!")
        # adding a column filling with all ones.(bias term)
        X = np.c_[X, np.ones(X.shape[0])]
        return X, y

##################################################

def load_data(data_path):
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
# These functions to generate artificial data. Not used?
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


# ======================================
