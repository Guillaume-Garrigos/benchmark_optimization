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
    def __init__(self, dataset=None):
        self.feature = None
        self.label = None
        self.name = ""
        self.path = ""
        self.nb_data = None
        self.dim = None
        self.config = get_config()

        # BEGIN : get access to the data
        if isinstance(dataset, str):
            # dataset is just the name of a dataset which is stored somewhere
            self.name = dataset
            X, y = self.load(dataset)
        elif isinstance(dataset, dict):
            # this is a dict of parameters that we can use to generate a sythetic dataset
            self.name = dataset.get('name', 'unspecified random')
            X, y = self.generate(dataset)
        else:
            raise Exception(f"Unknown dataset : {dataset}")
        # END : get access to the data

        X, y = self.preprocess_data(X, y) # normalize data etc
        # store what we have done
        n, d = X.shape
        self.feature = X
        self.label = y
        self.nb_data = n
        self.dim = d
        logging.info("Number of data points: {:d}; Number of features: {:d}".format(n, d))

    def load(self, dataset_name):
        """ load datasets from local storage in LibSVM format. """
        dataset_path_folder = self.config['problem'].get('dataset_path_folder', None)
        if dataset_path_folder is None:
            raise Exception("No path was given about where this dataset is stored. Can be set in the config file in problem > dataset_path_folder")
        data_path = os.path.join(dataset_path_folder, dataset_name) + '.txt'
        data = load_svmlight_file(data_path)
        return data[0].toarray(), data[1] # convert from scipy sparse matrix to dense if necessary
    
    def generate(self, param):
        # given a dict of parameters, return a generated dataset
        data_type = param.get('type', 100)
        nb_data = param.get('nb_data', 100)
        dim = param.get('dim', 50)
        error_level = param.get('error_level', 0.0)
        if data_type == 'phase retrieval':
            w = np.random.randn(dim)
            X = np.random.randn(nb_data, dim)
            noise = error_level * np.random.randn(nb_data)
            y = (X @ w)**2 + noise
        return X, y

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
        elif problem_type in ['regression','phase retrieval'] :
            X = X / np.max(np.abs(X), axis=0)  # scale features to [-1, 1]
        else:
            raise Exception("Unknown problem type!")
        # adding a column filling with all ones.(bias term)
        X = np.c_[X, np.ones(X.shape[0])]
        return X, y

##################################################




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
