import numpy as np
import os
import pickle
import copy



# A class to encode records. Updated during the running phase, exploited during the plotting phase
class Records():
    def __init__(self, name, solver):
        self.name = name
        self.solver_name = solver.name
        self.flavor_name = solver.flavor_name
        self.run_name = solver.run_name
        self.data_name = solver.problem.data.name
        self.value = []
        self.value_repetition = []
        self.param = copy.deepcopy(solver.param) # we need a copy here otherwise record_name will change everytime. We could have been smarter but so far doesn't seem to explode memory
        self.param.plot.set_record_name(name)
        self.do_we_save = solver.problem.param.do_we_save
    
    def store(self, solver): 
        # given a solver at a certain state, we compute some quantity and append it to Records.value
        pass

    def parent_solver(self, solver):
        self.solver_name = solver.name

    def save(self):
        if self.param.save_data:
            path = os.path.join(self.param.output_folder, self.run_name + '-' + self.name)
            with open(path, 'wb') as fp:
                pickle.dump(self, fp)
            return

    def process_values(self, time_value_repetition=None):
        """ Once recording values is done, we process them:
            take average, min, and max over the repetitions
        """
        param_plot = self.param.plot 
        result = self.value_repetition
        # result is a list of lists with different lengths
        self.value_avg = []
        self.value_min = []
        self.value_max = []
        self.xaxis = []
        # First we want all the lists to have the same lenght so we cut them if needed
        # At the same time we convert it to a numpy array, which will be easier than a list for plotting
        len_min = len(min(result, key=len)) # what is the smallest length
        result = np.array(list(map(lambda arr: arr[:len_min], result))) # cut the lists
        if time_value_repetition is not None:
            time_value_repetition = np.array(list(map(lambda arr: arr[:len_min], time_value_repetition)))
        # compute the average, min/max and variance
        self.value_avg = np.mean(result, axis=0)
        self.value_min = np.min(result, axis=0)
        self.value_max = np.max(result, axis=0)
        self.value_std = np.std(result, axis=0)
        # filter the values depending on the threshold parameter in the config file 
        # (if we don't want to plot values outside a certain range)
        # we decide that as soon as we are out of bounds we don't plot anymore
        # TODO : might not be a great idea but so far so good
        for i in range(self.value_avg.shape[0]):
            if (param_plot.threshold_down is not None and self.value_avg[i] < param_plot.threshold_down) or (param_plot.threshold_up is not None and self.value_avg[i] > param_plot.threshold_up):
                self.value_avg[i] = None
                self.value_min[i] = None
                self.value_max[i] = None
                self.value_std[i] = None
        # now define the xaxis (number of epochs)
        self.xaxis = np.arange(len_min)
        if time_value_repetition is not None:
            self.xaxis_time = np.mean(time_value_repetition, axis=0)
            self.xaxis_time = self.xaxis_time[:len_min]
        return
