import numpy as np
import logging
import os
import pickle
from datetime import datetime

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
        self.type = config['problem']['type']
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
        elif self.loss_name == "PhaseRetrieval":
            self.loss = loss.PhaseRetrieval()
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
        # summary
        self.summary = {'type' : self.type, 
                        'loss' : self.loss_name, 
                        'regularizer' : self.regularizer_name, 
                        'reg_parameter' : self.reg_parameter, 
                        'dataset_name' : self.data.name }
        # solution of the problem
        self.we_know_solution = False
        if config['problem']['save_solution'] or config['results']['use_solution'] or config['results']['subopt']: # subopt is legacy for use_solution
            self.solve()

    # Nice handles 
    # handy handles
        # annoying that functions evaluated at one point return not a float but a [float]
        # this is bc of the reshape. I think I do that to be able to compute function value on many vactors at once, but why? can't remember
        # tried to be efficient above, I hope it works. 
        # Basically using sum_i u_i u_i.T = U.T U where U=rows(u_i.T)
        """ The above could be easily modified if we want to work with autodiff
            in the end we only need access to functionand gradient handles.
            The switch between one kind of problem or the other should be handled 
            at the init of Problem(), certainly with some optional parameter
        """
    def function_sampled(self, x, i): 
        return self.loss.val(self.data.label[i], self.data.feature[i, :] @ x) + self.reg_parameter * self.regularizer.val(x)
    def gradient_sampled(self, x, i): 
        return self.loss.prime(self.data.label[i], self.data.feature[i, :] @ x) * self.data.feature[i, :] + self.reg_parameter * self.regularizer.prime(x)
    def hessian_sampled(self, x, i): 
        return self.loss.dprime(self.data.label[i], self.data.feature[i, :] @ x) * self.data.feature[i, :].reshape((-1,1)) @ self.data.feature[i, :].reshape((1,-1)) + self.reg_parameter * self.regularizer.dprime(x)
    def function(self, x): 
        return np.mean(self.loss.val(self.data.label, self.data.feature @ x).reshape(-1, 1).flatten(), axis=0) + self.reg_parameter * self.regularizer.val(x)
    def gradient(self, x): 
        return self.data.feature.T @ self.loss.prime(self.data.label, self.data.feature @ x) / self.nb_data + self.reg_parameter * self.regularizer.prime(x)
    def hessian(self, x):
        # Hessian at x is : 
        # (1/n) * Phi.T @ Diag(loss'') @ Phi
        # we write it as (1/n) h.T @ h with h = Diag(sqrt(loss'')) @ Phi
        # and use the fact that Diag(v) @ A = v * A in python
        h = np.sqrt(self.loss.dprime(self.data.label, self.data.feature @ x)).reshape(-1, 1) * self.data.feature
        return (h.T @ h)/self.nb_data + self.reg_parameter * self.regularizer.dprime(x)
        
    def solve(self):
        # gets the "exact" solution of the problem
        solution = {}
        new_solution = False
        if self.param.config['problem']['load_solution']:
            solution = self.load_solution()
        if solution == {}: # load failed or we don't want to load
            solution = self.compute_solution()
            new_solution = True
        # we check the solution is good
        g_0 = self.gradient(np.zeros(self.dim))
        if solution['gradient_norm'] < 1e-5*np.sqrt(g_0 @ g_0):
            pass
        else: # failure
            print("The gradient at given optimum is relatively larger than 1e-5, we think it is not an optimum")
            solution = {}
        # store it in the Problem class
        if solution != {}: 
            self.we_know_solution = True
            self.solution_info = solution['solver_info']
            self.optimal_solution = solution['minimizer']
            self.optimal_value = solution['infimum']
            self.optimal_value_sampled = [self.function_sampled(solution['minimizer'], i) for i in range(self.nb_data)]
            self.optimal_gradient_sqnorm_sampled = [np.linalg.norm(self.gradient_sampled(solution['minimizer'], i))**2 for i in range(self.nb_data)]
            # save the solution if we want ; and if it wasn't loaded already
            if self.param.config['problem'].get('save_solution', False) and new_solution:
                self.save_solution(solution)
        return
    
    def problem_file_name(self):
        # Given a problem,
        # returns a unique string "characterizing" the problem
        # dataset name + problem type + loss name + regularizer name
        # we don't add the regularization parameter which could be messy
        return self.data.name +'-'+ self.type +'-'+ self.loss_name +'-'+ self.regularizer_name
    
    def save_solution(self, solution):
        timestamp = datetime.today().strftime('%y%m%d_%H%M%S')
        solution['timestamp'] = timestamp
        solution['signature'] = 'solution_solver'
        folder = self.param.config['problem']['solution_path_folder']
        if not os.path.exists(folder): 
            os.makedirs(folder)
        filename = self.problem_file_name() +'-'+ timestamp
        path = os.path.join(folder, filename)
        with open(path, 'wb') as file:
            pickle.dump(solution, file)
        return
    
    def compute_solution(self):
        # try to compute the exact solution of the problem
        # returns a dict with all the information if computation is successful
        # return empty dict otherwise
        if self.loss_name == "Logistic" and self.regularizer_name == "L2":
            print("Solve the problem with scipy fmin_l_bfgs_b")
            # TODO review this bc i don't understand why we complicate things with utils.stuff... Maybe inherited from Jiabin?
            pgtol = self.param.config['problem']['pgtol']
            factr = float(self.param.config['problem']['factr'])
            optimum, f_opt, info = fmin_l_bfgs_b(
                func=utils.f_val_logistic, 
                x0=np.zeros(self.dim), 
                fprime=utils.f_grad_logistic,
                args=(self.data.feature, self.data.label, self.loss, self.regularizer, self.reg_parameter), 
                pgtol = pgtol,
                factr = factr)
            # we store the results
            solution = {'minimizer' : optimum, 
                        'infimum' : f_opt, 
                        'gradient_norm' : np.linalg.norm(info.pop('grad')), # pop deletes 'grad' after use, saving memory 
                        'solver_info' : {**info, 'pgtol' : pgtol, 'factr' : factr}, # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html#scipy.optimize.fmin_l_bfgs_b
                        'problem' : self.summary }
            return solution
        else: # TODO
            print("The problem can't be solved (not yet implemented)")
            return {}
            
    
    def load_solution(self):
        # if it exists, load the solution of the problem at hand
        # returns a dict (contains the solution and stuff) eventually empty
        folder = self.param.config['problem']['solution_path_folder']
        if os.path.exists(folder):
            for file in os.scandir(folder):
                with open(file.path, 'rb') as file:
                    solution = pickle.load(file)
                    if isinstance(solution, dict) and solution.get('signature') == 'solution_solver': # just in case there are other files in there
                        # is this the solution of our problem at hand?
                        if solution['problem'] == self.summary:
                            return solution
        return {}
    
    def get_gradient_variance(self, batch_size=1):
        if batch_size == 1:
            return np.average( self.optimal_gradient_sqnorm_sampled )
    
    def expected_smoothness(self):
        if self.loss_name == "L2":
            return utils.max_Li_ridge(self.data.feature, self.reg_parameter)
        elif self.loss_name == "Logistic":
            return utils.max_Li_logistic(self.data.feature, self.reg_parameter)
        elif self.loss_name == "PhaseRetrieval":
            return 1.0
        else:
            print(f"Warning: The loss {self.loss_name} is unknown. The expected smoothness constant couldn't be defined")
            return 1.0
##################################################