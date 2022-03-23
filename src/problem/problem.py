import numpy as np
import logging

from src.parameters import Parameters
import config
import src.problem.loss as loss
import src.problem.regularizer as regularizer
from src import utils
from scipy.optimize import fmin_l_bfgs_b

""" The class Problem() should contain everything needed to define the problem
    Contains the loss, regularizer and reg.parameter, and provides easy handles to function/gradient
    Contains the data via the class Data()
    Contains all the parameters needed to run the solvers and plot/save the results
    It is here that we eventually call an external routine to solve the problem
        useful if we want to compare to a baseline solution

    Right now all our problems depend on Data(), so Problems() is init with Data()
        This could be changed in the future (for non data-driven problems)
"""


##################################################
class Problem():
    # stores the main component of the problem to solve
    def __init__(self, data):
        # parameters
        self.param = Parameters(data)
        # data
        self.data = data
        self.nb_data = data.nb_data
        self.dim = data.dim
        # names
        self.loss_name = config.loss
        self.regularizer_name = config.regularizer
        # loss
        if self.loss_name == "L2":
            self.loss = loss.L2()
        elif self.loss_name == "PseudoHuber":
            self.loss = loss.PseudoHuberLoss(delta=1.0)
        elif self.loss_name == "Logistic":
            self.loss = loss.LogisticLoss()
        else:
            raise Exception("Unknown loss function!")
        # regularizer
        if self.regularizer_name == "L2":
            self.regularizer = regularizer.L2()
        elif self.regularizer_name == "PseudoHuber":
            self.regularizer = regularizer.PseudoHuber(delta=1.0)
        else:
            raise Exception("Unknown regularizer type!")
        # regularization parameter
        if isinstance(config.reg_parameter, float):
            self.reg_parameter = config.reg_parameter
        else: # we assume that it is a lambda function of nb_data
            self.reg_parameter = config.reg_parameter(self.nb_data)
        logging.info("Regularization param: {}".format(self.reg_parameter))
        # solution of the problem
        self.solve()
        # handy handles
        self.function_sampled = lambda x, i: self.loss.val(self.data.label[i], self.data.feature[i, :] @ x) + self.reg_parameter * self.regularizer.val(x)
        self.gradient_sampled = lambda x, i: self.loss.prime(self.data.label[i], self.data.feature[i, :] @ x) * self.data.feature[i, :] + self.reg_parameter * self.regularizer.prime(x)
        self.function = lambda x: np.mean(self.loss.val(self.data.label, self.data.feature @ x).reshape(-1, 1), axis=0) + self.reg_parameter * self.regularizer.val(x)
        self.gradient = lambda x: np.mean(self.loss.prime(self.data.label, self.data.feature @ x).reshape(-1, 1) * self.data.feature, axis=0) + self.reg_parameter * self.regularizer.prime(x)
        """ The above could be easily modified if we want to work with autodiff
            in the end we only need access to functionand gradient handles.
            The switch between one kind of problem or the other should be handled 
            at the init of Problem(), certainly with some optional parameter
        """


    def solve(self):
        # try to obtain the optimum by using l-bfgs method
        # but only the L2 logistic regression is supported so far
        f_opt = 0.0
        optimum = None
        self.we_know_solution = False

        if config.subopt and self.loss_name == "Logistic" and self.regularizer_name == "L2":
            print("Solve the problem with scipy fmin_l_bfgs_b")
            optimum, f_opt, d_info = fmin_l_bfgs_b(
                func=utils.f_val_logistic, x0=np.zeros(self.dim), fprime=utils.f_grad_logistic,
                args=(self.data.feature, self.data.label, self.loss, self.regularizer, self.reg_parameter), pgtol=1e-07)

            # verify that the gradient at optimum is close to zero
            g_opt = d_info['grad']
            if np.sqrt((g_opt @ g_opt)) > 1e-5:
                print("The gradient at given optimum is larger than 1e-5, we think it is not an optimum")
            else: # its ok
                self.optimal_value = f_opt
                self.optimal_solution = optimum
                self.we_know_solution = True
        return
##################################################