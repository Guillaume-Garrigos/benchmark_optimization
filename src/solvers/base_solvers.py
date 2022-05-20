import numpy as np
from src import utils

from src.solvers.solver import Solver

# ==================================
# Baseline methods
# ==================================
class SAG(Solver):
    """
    Stochastic average gradient algorithm.
    """
    name = "SAG"

    def __init__(self, problem):
        Solver.__init__(self, problem)
        self.append_records("gradient_norm", "function_value")
        self.set_learning_rate()
        # Old gradients
        self.gradient_memory = None
        self.y = None

    def initialization_variables(self):
        Solver.initialization_variables(self)
        n, d = self.problem.nb_data, self.problem.dim
        self.gradient_memory = np.zeros((n, d))
        self.y = np.zeros(d)
        return

    def set_learning_rate(self):
        if self.problem.loss_name == "L2":
            self.lr = self.param.lr / utils.max_Li_ridge(self.problem.data.feature, self.problem.reg_parameter)
        elif self.problem.loss_name == "Logistic":
            self.lr = self.param.lr / utils.max_Li_logistic(self.problem.data.feature, self.problem.reg_parameter)
        else:
            print("Warning!!!")
            self.lr = 0.01

    def run_epoch(self):
        feature = self.problem.data.feature
        label = self.problem.data.label
        reg = self.problem.reg_parameter
        n = self.problem.nb_data
        iis = np.random.randint(low=0, high=n, size=n)
        for i in iis:
            # gradient of i-th loss
            grad_i = self.loss.prime(label[i], feature[i, :] @ self.x) * feature[i, :] \
                     + reg * self.regularizer.prime(self.x)
            # update
            self.y += grad_i - self.gradient_memory[i]
            self.x -= self.lr * self.y / n
            self.gradient_memory[i] = grad_i
        return
    
class SGD(Solver):

    name = "SGD"
    def __init__(self, problem):
        Solver.__init__(self, problem)
        self.append_records("gradient_norm", "function_value", "stepsize")
        self.set_learning_rate()
        self.lr = self.lr * 0.25
        self.stepsize = self.lr

    def initialization_variables(self):
        Solver.initialization_variables(self)
        self.extrapolation_parameter = 1

    def set_learning_rate(self):
        if self.problem.loss_name == "L2":
            self.lr = self.param.lr / utils.max_Li_ridge(self.problem.data.feature, self.problem.reg_parameter)
        elif self.problem.loss_name == "Logistic":
            self.lr = self.param.lr / utils.max_Li_logistic(self.problem.data.feature, self.problem.reg_parameter)
        else:
            print("Warning!!!")
            self.lr = 0.01

    def run_epoch(self):
        feature = self.problem.data.feature
        label = self.problem.data.label
        reg = self.problem.reg_parameter
        n = self.problem.nb_data
        iis = np.random.randint(low=0, high=n, size=n)  # uniform sampling
        for i in iis:
            grad_loss = self.loss.prime(label[i], feature[i, :] @ self.x) * feature[i, :] 
            grad_reg = self.regularizer.prime(self.x)
            grad = grad_loss + reg * grad_reg
            xhat = self.x - self.lr * grad
            self.x = (1-self.extrapolation_parameter)*self.x + self.extrapolation_parameter*xhat
        return
    


class SVRG(Solver):
    """
    Stochastic variance reduction gradient algorithm.

    reference: Accelerating Stochastic Gradient Descent using Predictive Variance Reduction, Johnson & Zhang

    Note: for all stochastic methods, we measure the performance as a function of the number of effective passes
    through the data, measured as the number of queries to access single gradient (or Hessian) divided by
    the size of dataset. To have a fair comparison with others methods, for SVRG, we pay a careful attention
    to the step where we do a full pass of dataset at the reference point,
    it means that the effective passes should be added one after this step.
    """

    name = "SVRG"
    def __init__(self, problem):
        Solver.__init__(self, problem)
        self.append_records("gradient_norm", "function_value")
        self.set_learning_rate()
        self.x_ref = None
        self.grad_ref = None
        self.do_inner_loop = False

    def initialization_variables(self):
        Solver.initialization_variables(self)

    def set_learning_rate(self):
        if self.problem.loss_name == "L2":
            self.lr = self.param.lr / utils.max_Li_ridge(self.problem.data.feature, self.problem.reg_parameter)
        elif self.problem.loss_name == "Logistic":
            self.lr = self.param.lr / utils.max_Li_logistic(self.problem.data.feature, self.problem.reg_parameter)
        else:
            print("Warning!!!")
            self.lr = 0.01

    def _full_pass(self):
        feature = self.problem.data.feature
        label = self.problem.data.label
        reg = self.problem.reg_parameter
        self.x_ref = self.x.copy()
        self.grad_ref = np.mean(self.loss.prime(label, feature @ self.x).reshape(-1, 1) * feature, axis=0) + \
                        reg * self.regularizer.prime(self.x)
        self.x -= self.lr * self.grad_ref
        return

    def _inner_loop(self):
        feature = self.problem.data.feature
        label = self.problem.data.label
        reg = self.problem.reg_parameter
        n = self.problem.nb_data
        iis = np.random.randint(low=0, high=n, size=n)
        for i in iis:
            grad_i = self.loss.prime(label[i], feature[i, :] @ self.x) * feature[i, :] \
                     + reg * self.regularizer.prime(self.x)
            grad_i_ref = self.loss.prime(label[i], feature[i, :] @ self.x_ref) * feature[i, :] \
                         + reg * self.regularizer.prime(self.x_ref)
            d_i = grad_i - grad_i_ref + self.grad_ref
            self.x -= self.lr * d_i
        return

    def run_epoch(self):
        if self.do_inner_loop:
            self._inner_loop()
            self.do_inner_loop = False
        else:
            self._full_pass()
            self.do_inner_loop = True
        return


class Adam(Solver):

    name = "Adam"
    def __init__(self, problem):
        Solver.__init__(self, problem)
        self.append_records("gradient_norm", "function_value")
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.update_cnt = 0
        self.m = None
        self.v = None

    def initialization_variables(self):
        Solver.initialization_variables(self)
        d = self.problem.dim
        self.m = np.zeros(d)
        self.v = np.zeros(d)
        return

    def run_epoch(self):
        feature = self.problem.data.feature
        label = self.problem.data.label
        reg = self.problem.reg_parameter
        n = self.problem.nb_data
        iis = np.random.randint(low=0, high=n, size=n)
        for i in iis:
            self.update_cnt += 1
            grad_i = self.loss.prime(label[i], feature[i, :] @ self.x) * feature[i, :] \
                     + reg * self.regularizer.prime(self.x)
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_i
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_i * grad_i)
            m_hat = self.m / (1 - self.beta1 ** self.update_cnt)
            v_hat = self.v / (1 - self.beta2 ** self.update_cnt)
            direction = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.x -= direction  # update
        return


#########################
# Deterministic Algorithm
#########################
class GD(Solver):

    name = "GD"
    def __init__(self, problem):
        Solver.__init__(self, problem)
        self.append_records("gradient_norm", "function_value")
        self.set_learning_rate()

    def initialization_variables(self):
        Solver.initialization_variables(self)

    def set_learning_rate(self):
        if self.problem.loss_name == "L2" and self.problem.regularizer_name == 'L2':
            self.lr = self.param.lr / utils.lipschitz_ridge(self.problem.data.feature, self.problem.reg_parameter)
        elif self.problem.loss_name == "Logistic" and self.problem.regularizer_name == 'L2':
            self.lr = self.param.lr / utils.lipschitz_logistic(self.problem.data.feature, self.problem.reg_parameter)
        else:
            print("Warning!!! GD learning rate")
            self.lr = 0.01

    def run_epoch(self):
        feature = self.problem.data.feature
        label = self.problem.data.label
        reg = self.problem.reg_parameter
        grad = np.mean(self.loss.prime(label, feature @ self.x).reshape(-1, 1) * feature, axis=0) \
               + reg * self.regularizer.prime(self.x)
        self.x -= self.lr * grad
        return


class Newton(Solver):

    name = "Newton"
    def __init__(self, problem):
        Solver.__init__(self, problem)
        self.append_records("gradient_norm", "function_value")
        self.lr = 1.0

    def initialization_variables(self):
        Solver.initialization_variables(self)

    def run_epoch(self):
        feature = self.problem.data.feature
        label = self.problem.data.label
        reg = self.problem.reg_parameter
        n = self.problem.nb_data
        grad = np.mean(self.loss.prime(label, feature @ self.x).reshape(-1, 1) * feature, axis=0) \
               + reg * self.regularizer.prime(self.x)
        h = np.sqrt(self.loss.dprime(label, feature @ self.x)).reshape(-1, 1) * feature
        hess = reg * np.diag(self.regularizer.dprime(self.x)) + (h.T @ h) / n
        self.x -= self.lr * np.linalg.solve(hess, grad)
        return
