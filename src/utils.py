import numpy as np


def lipschitz_ridge(X, reg):
    n, d = X.shape
    return np.linalg.norm(X, ord=2) ** 2 / n + reg


def lipschitz_logistic(X, reg):
    n, d = X.shape
    return np.linalg.norm(X, ord=2) ** 2 / (4. * n) + reg


def max_Li_ridge(X, reg):
    return np.max(np.sum(X ** 2, axis=1)) + reg


def max_Li_logistic(X, reg):
    return 0.25 * np.max(np.sum(X ** 2, axis=1)) + reg



def cond(data):
    # data: numpy array of shape n x d
    # singular val of (data.T @ data)
    # return condition number of a given dataset, i.e., max(singular val) / min non zero(singular val)
    n, d = data.shape
    if n >= d:
        s = np.linalg.svd(data.T@data, compute_uv=False, hermitian=True)
    if n < d:
        s = np.linalg.svd(data@data.T, compute_uv=False, hermitian=True)
    # singular values are sorted in descending order
    min_non_zero_pos = len(s) - 1
    while min_non_zero_pos >= 0 and s[min_non_zero_pos] == 0:
        min_non_zero_pos -= 1
    if min_non_zero_pos < 0:
        return float('inf')
    return s[0] / s[min_non_zero_pos]


# ==========================
# two auxiliary functions to incorporate the use of scipy fmin_l_bfgs_b
# ==========================
def f_val_logistic(x, data, label, loss, regularizer, reg):
    return np.mean(loss.val(label, data @ x)) + reg * regularizer.val(x)


def f_grad_logistic(x, data, label, loss, regularizer, reg):
    return np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
