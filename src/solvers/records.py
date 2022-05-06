import numpy as np
import os
import pickle
import copy

import config


# A class to encode records. Updated during the running phase, exploited during the plotting phase
class Records():
    def __init__(self, name, solver):
        self.name = name
        self.solver_name = solver.name
        self.data_name = solver.problem.data.name
        self.value = []
        self.value_repetition = []
        self.param = copy.deepcopy(solver.param) # we need a copy here otherwise record_name will change everytime. We could have been smarter but so far doesn't seem to explode memory
        self.param.plot.set_record_name(name)
        self.do_we_save = config.save
    
    def store(self, solver): 
        # given a solver at a certain state, we compute some quantity and append it to Records.value
        pass

    def parent_solver(self, solver):
        self.solver_name = solver.name

    def save(self):
        if self.do_we_save:
            path = os.path.join(self.param.output_folder, self.solver_name + '-' + self.name)
            with open(path, 'wb') as fp:
                pickle.dump(self, fp)
            return

    def process_values(self, time_value_repetition=None):
        """ Once recording values is done, we process them:
            take average, min, and max over the repetitions
        """
        param = self.param.plot 
        result = self.value_repetition
        # result is a list of lists with different lengths
        self.value_avg = []
        self.value_min = []
        self.value_max = []
        self.xaxis = []
        # cut it with min_len and convert it to numpy array, needed for plot
        len_min = len(min(result, key=len)) # smallest length
        result = np.array(list(map(lambda arr: arr[:len_min], result)))
        # compute the average; maybe cut it when under threshold; check positivity
        self.value_avg = np.mean(result, axis=0)
        if param.threshold is not None:
            threshold = param.threshold
            len_cut = np.argmax(self.value_avg <= threshold) + \
                1 if np.sum(self.value_avg <= threshold) > 0 else len(self.value_avg)
        else:
            len_cut = len(self.value_avg)
        self.value_avg = self.value_avg[:len_cut]
        if (self.value_avg < 0).any(): # this to be IMPROVED
            print("WARNING: there are negative numbers in sub-optimality plots !!!")
            print("WARNING: replace all negative numbers by absolute in in sub-optimality plots !!!")
            self.value_avg = np.abs(self.value_avg)
        # compute min/max for displaying variance
        self.value_min = np.min(result, axis=0)[:len(self.value_avg)]
        self.value_max = np.max(result, axis=0)[:len(self.value_avg)]
        # now define the xaxis (number of epochs)
        self.xaxis = np.arange(len(self.value_avg))
        if time_value_repetition is not None:
            time_value_repetition = np.array(list(map(lambda arr: arr[:len_min], time_value_repetition)))
            self.xaxis_time = np.mean(time_value_repetition, axis=0)
            self.xaxis_time = self.xaxis_time[:len_cut]
        return

# ====================================
# Basic records used by most solvers
# ====================================

class Gradient_norm(Records):
    name = "gradient_norm"
    def __init__(self, solver):
        Records.__init__(self, self.name, solver)
        self.param.plot.xlabel = "Effective Passes"
        self.param.plot.ylabel = r"$\| \nabla f \|$"
    
    def store(self, solver):
        feature = solver.problem.data.feature
        label = solver.problem.data.label
        reg = solver.problem.reg_parameter
        g = np.mean(solver.loss.prime(label, feature @ solver.x).reshape(-1, 1) * feature, axis=0) \
            + reg * solver.regularizer.prime(solver.x)
        self.value.append(np.sqrt(g @ g))

class Function_value(Records):
    name = "function_value"
    def __init__(self, solver):
        Records.__init__(self, self.name, solver)
        self.param.plot.xlabel = "Effective Passes"
        self.param.plot.ylabel = r"$f - f^*$"
    
    def store(self, solver):
        feature = solver.problem.data.feature
        label = solver.problem.data.label
        reg = solver.problem.reg_parameter
        fval = np.mean(solver.loss.val(label, feature @ solver.x)) + reg * solver.regularizer.val(solver.x)
        if solver.problem.we_know_solution:
            fval = fval - solver.problem.optimal_value
        self.value.append(fval)

class Time_epoch(Records):
    name = "time_epoch"
    def __init__(self, solver):
        Records.__init__(self, self.name, solver)
        self.param.plot.xlabel = "Effective Passes"
        self.param.plot.ylabel = "Time (s)"
        self.param.plot.threshold = None # we don't want to threshold that
    
    def store(self, solver):
        self.value.append(solver.total_running_time)
