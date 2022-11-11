import numpy as np
import logging

from src.parameters import Parameters
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
        config = self.param.config
        # data
        self.data = data
        self.nb_data = data.nb_data
        self.dim = data.dim
        # names
        self.loss_name = config['problem']['loss']
        self.regularizer_name = config['problem']['regularizer']
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
        if isinstance(config['problem']['reg_parameter'], float):
            self.reg_parameter = config['problem']['reg_parameter']
        else: # we assume it is a string
            if config['problem']['reg_parameter'] == "inv_sqrt_data":
                self.reg_parameter = 1./np.sqrt(self.nb_data)
        logging.info("Regularization param: {}".format(self.reg_parameter))
        # handy handles
        # annoying that functions evaluated at one point return not a float but a [float]
        # this is bc of the reshape. I think I do that to be able to compute function value on many vactors at once, but why? can't remember
        self.function_sampled = lambda x, i: self.loss.val(self.data.label[i], self.data.feature[i, :] @ x) + self.reg_parameter * self.regularizer.val(x)
        self.gradient_sampled = lambda x, i: self.loss.prime(self.data.label[i], self.data.feature[i, :] @ x) * self.data.feature[i, :] + self.reg_parameter * self.regularizer.prime(x)
        self.hessian_sampled = lambda x, i: self.loss.dprime(self.data.label[i], self.data.feature[i, :] @ x) * self.data.feature[i, :].reshape((-1,1)) @ self.data.feature[i, :].reshape((1,-1)) + self.reg_parameter * self.regularizer.dprime(x)
        self.function = lambda x: np.mean(self.loss.val(self.data.label, self.data.feature @ x).reshape(-1, 1).flatten(), axis=0) + self.reg_parameter * self.regularizer.val(x)
        self.gradient = lambda x: np.mean(self.loss.prime(self.data.label, self.data.feature @ x).reshape(-1, 1) * self.data.feature, axis=0) + self.reg_parameter * self.regularizer.prime(x)
        self.hessian = lambda x: self.data.feature.T @ ( self.loss.dprime(self.data.label, self.data.feature @ x).reshape(-1, 1) * self.data.feature )/self.nb_data + self.reg_parameter * self.regularizer.dprime(x)
        # tried to be efficient above, I hope it works. 
        # Basically using sum_i u_i u_i.T = U.T U where U=rows(u_i.T)
        """ The above could be easily modified if we want to work with autodiff
            in the end we only need access to functionand gradient handles.
            The switch between one kind of problem or the other should be handled 
            at the init of Problem(), certainly with some optional parameter
        """
        # solution of the problem
        self.we_know_solution = False
        if config['results']['subopt']:
            self.solve()


    def solve(self):
        # try to obtain the optimum by using l-bfgs method
        # but only the L2 logistic regression is supported so far
        f_opt = 0.0
        optimum = None
        #self.optimal_value = None
        if self.loss_name == "Logistic" and self.regularizer_name == "L2":
            print("Solve the problem with scipy fmin_l_bfgs_b")
            optimum, f_opt, d_info = fmin_l_bfgs_b(
                func=utils.f_val_logistic, x0=np.zeros(self.dim), fprime=utils.f_grad_logistic,
                args=(self.data.feature, self.data.label, self.loss, self.regularizer, self.reg_parameter), pgtol=1e-07)

            # verify that the gradient at optimum is close to zero
            g_opt = d_info['grad']
            g_0 = self.gradient(np.zeros(self.dim))
            if np.sqrt((g_opt @ g_opt)) > 1e-5*np.sqrt(g_0 @ g_0):
                print("The gradient at given optimum is relatively larger than 1e-5, we think it is not an optimum")
                # here self.we_know_solution remains equal to False
                # this is what should be tested later in the code
            else: # its ok
                self.we_know_solution = True
                self.optimal_solution = optimum
                self.optimal_value = f_opt
                self.optimal_value_sampled = [self.function_sampled(optimum, i) for i in range(self.nb_data)]
        return
    
    def expected_smoothness(self):
        if self.loss_name == "L2":
            return utils.max_Li_ridge(self.data.feature, self.reg_parameter)
        elif self.loss_name == "Logistic":
            return utils.max_Li_logistic(self.data.feature, self.reg_parameter)
        else:
            print(f"Warning: The loss {self.loss_name} is unknown. The expected smoothness constant couldn't be defined")
            return 1.0
##################################################