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
            path = os.path.join(self.param.output_folder, self.flavor_name + '-' + self.name)
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
        if param.threshold_down != 0.0 and param.threshold_down is not None:
            threshold_down = param.threshold_down
            len_cut = np.argmax(self.value_avg <= threshold_down) + \
                1 if np.sum(self.value_avg <= threshold_down) > 0 else len(self.value_avg)
        else:
            len_cut = len(self.value_avg)
        self.value_avg = self.value_avg[:len_cut]
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
