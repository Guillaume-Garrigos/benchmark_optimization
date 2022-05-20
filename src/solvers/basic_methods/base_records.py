import numpy as np
from src.solvers import Records


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
        
        
# New record : the stepsize (whatever it is as long it is called .stepsize)
# maybe we should have a generic way to plot a method of solver which is a float ..?
class Stepsize(Records):
    name = "stepsize"
    def __init__(self, solver):
        Records.__init__(self, self.name, solver)
        self.param.plot.xlabel = "Effective Passes"
        self.param.plot.ylabel = "Stepsize"
        self.param.plot.threshold = None # we don't want to threshold that
    
    def store(self, solver):
        self.value.append(solver.stepsize)
