import numpy as np
from src import utils

from src.solvers import Solver

# ==================================
# Baseline methods
# ==================================
class SGD(Solver):
    name = "SGD"
    def __init__(self, problem, args={}):
        Solver.__init__(self, problem, args=args)
        self.append_records("gradient_norm", "function_value", "stepsize")

    def run_epoch(self):
        n = self.problem.nb_data
        sampling = np.random.randint(low=0, high=n, size=n)  # uniform sampling
        for i in sampling:
            grad_i_x = self.problem.gradient_sampled(self.x, i)
            self.stepsize = self.get_stepsize()
            x_new = self.x - self.stepsize * grad_i_x
            self.x = (1-self.extrapolation_parameter)*self.x + self.extrapolation_parameter*x_new
            self.increment_iteration()
        return

class SAG(Solver):
    """
    Stochastic average gradient algorithm.
    """
    name = "SAG"

    def __init__(self, problem, args={}):
        Solver.__init__(self, problem, args=args)
        self.append_records("gradient_norm", "function_value")
        # Old gradients
        self.gradient_memory = None
        self.y = None

    def initialization_variables(self):
        Solver.initialization_variables(self)
        n, d = self.problem.nb_data, self.problem.dim
        self.gradient_memory = np.zeros((n, d))
        self.y = np.zeros(d)
        self.stepsize = self.get_stepsize()
        return

    def run_epoch(self):
        n = self.problem.nb_data
        sampling = np.random.randint(low=0, high=n, size=n)
        for i in sampling:
            self.stepsize = self.get_stepsize()
            # gradient of i-th loss
            grad_i_x = self.problem.gradient_sampled(self.x, i)
            # update
            self.y += grad_i_x - self.gradient_memory[i]
            self.x = self.x - self.stepsize * self.y / n
            self.gradient_memory[i] = grad_i_x
            self.increment_iteration()
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
    def __init__(self, problem, args={}):
        Solver.__init__(self, problem, args=args)
        self.append_records("gradient_norm", "function_value")
        self.x_ref = None
        self.grad_ref = None
        self.do_inner_loop = False
    
    def initialization_variables(self):
        Solver.initialization_variables(self)
        self.stepsize = self.get_stepsize()

    def run_full_pass(self):
        self.x_ref = self.x.copy()
        self.grad_ref = self.problem.gradient(self.x)
        self.x = self.x - self.stepsize * self.grad_ref
        self.increment_iteration()
        return

    def run_inner_loop(self):
        n = self.problem.nb_data
        iis = np.random.randint(low=0, high=n, size=n)
        for i in iis:
            self.stepsize = self.get_stepsize()
            grad_i_x = self.problem.gradient_sampled(self.x, i)
            grad_i_xref = self.problem.gradient_sampled(self.x_ref, i)
            d_i = grad_i_x - grad_i_xref + self.grad_ref
            self.x = self.x - self.stepsize * d_i
            self.increment_iteration()
        return

    def run_epoch(self):
        if self.do_inner_loop:
            self.run_inner_loop()
            self.do_inner_loop = False
        else:
            self.run_full_pass()
            self.do_inner_loop = True
        return


class Adam(Solver):

    name = "Adam"
    def __init__(self, problem, args={}):
        Solver.__init__(self, problem, args=args)
        self.append_records("gradient_norm", "function_value")
        self.stepsize = 0.001
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
        n = self.problem.nb_data
        iis = np.random.randint(low=0, high=n, size=n)
        for i in iis:
            self.update_cnt += 1
            grad_i_x = self.problem.gradient_sampled(self.x, i)
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_i_x
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_i_x * grad_i_x)
            m_hat = self.m / (1 - self.beta1 ** self.update_cnt)
            v_hat = self.v / (1 - self.beta2 ** self.update_cnt)
            direction = m_hat / (np.sqrt(v_hat) + self.eps)
            self.x = self.x - self.stepsize * direction  # update
        return


#########################
# Deterministic Algorithm
#########################
class GD(Solver):

    name = "GD"
    def __init__(self, problem, args={}):
        Solver.__init__(self, problem, args=args)
        self.append_records("gradient_norm", "function_value")

    def initialization_variables(self):
        Solver.initialization_variables(self)
        self.stepsize = self.get_stepsize()

    def run_epoch(self):
        self.stepsize = self.get_stepsize()
        grad_x = self.problem.gradient(self.x)
        self.x = self.x - self.stepsize * grad_x
        self.increment_iteration()
        return

class Newton(Solver):

    name = "Newton"
    def __init__(self, problem, args={}):
        Solver.__init__(self, problem, args=args)
        self.append_records("gradient_norm", "function_value")
        self.stepsize = 1.0

    def initialization_variables(self):
        Solver.initialization_variables(self)

    def run_epoch(self):
        feature = self.problem.data.feature
        label = self.problem.data.label
        reg = self.problem.reg_parameter
        n = self.problem.nb_data
        grad_x = self.problem.gradient(self.x)
        h = np.sqrt(self.loss.dprime(label, feature @ self.x)).reshape(-1, 1) * feature
        hess = reg * (self.regularizer.dprime(self.x)) + (h.T @ h) / n
        self.x -= self.stepsize * np.linalg.solve(hess, grad_x)
        return
