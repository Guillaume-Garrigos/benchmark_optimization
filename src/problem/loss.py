import numpy as np

def smart_inv_exp(x):
    # given a number x, returns 1/(1+e^-x)
    # make sure that we don't manipulate large numbers
    # if x is an array, applies component-wise
    # this function has intersting properties (lets note f(x)) :
    # 1/f(x) = 1+e^-x
    # 1-f(x) = 1/(1+e^x)
    # f(-x) = e^x/(1+e^x)
    try: # we hope x is an array
        output = np.zeros(x.shape)
    except:
        print("trying to compute the logistic function on a float")
    idx = x > 0
    output[idx] = 1.0/(1 + np.exp(-x[idx]))
    exp_smol = np.exp(x[~idx])
    output[~idx] = exp_smol/(1 + exp_smol)
    return output

class LogisticLoss:
    @staticmethod
    def val(y, y_hat):
        #return np.log(1 + np.exp(-y * y_hat))
        return -np.log( smart_inv_exp(y * y_hat)) # use 1/f(x) = 1+e^-x

    @staticmethod
    def prime(y, y_hat):
        #return -y / (1 + np.exp(y * y_hat))
        return -y * (1 - smart_inv_exp(y * y_hat)) # use 1-f(x) = 1/(1+e^x)

    @staticmethod
    def dprime(y, y_hat):
        #a = np.exp(y * y_hat)
        #return a / ((1 + a) ** 2)
        # instead we use f(-x)*(1-f(x)) = e^x/(1+e^x) * 1/(1+e^x)
        return smart_inv_exp(-y * y_hat)*(1 - smart_inv_exp(y * y_hat))


class L2:
    @staticmethod
    def val(y, y_hat):
        return (y - y_hat)**2 / 2.

    @staticmethod
    def prime(y, y_hat):
        return y_hat - y

    @staticmethod
    def dprime(y, y_hat):
        return np.ones_like(y_hat)


class PseudoHuberLoss:

    def __init__(self, delta=1.0):
        self.delta = delta

    def val(self, y, y_hat):
        diff = y_hat - y
        return (self.delta ** 2) * (np.sqrt(1. + (diff / self.delta) ** 2) - 1.)

    def prime(self, y, y_hat):
        diff = y_hat - y
        return diff / np.sqrt(1. + (diff / self.delta) ** 2)

    def dprime(self, y, y_hat):
        diff = y_hat - y
        return np.power((1. + (diff / self.delta) ** 2), -1.5)


class PhaseRetrieval:
    def __init__(self):
        pass
    def val(self, y, y_hat):
        return np.abs(y_hat**2-y**2)
    def prime(self, y, y_hat):
        return 2*y_hat*np.sign(y_hat**2 - y**2)
    def dprime(self, y, y_hat):
        return 2*np.sign(y_hat**2 - y**2) 
    
        