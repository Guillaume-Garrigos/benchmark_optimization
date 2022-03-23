import matplotlib.pyplot as plt
import os
import pickle

from src.parameters import Parameters
from src.solvers.records import Records


def update_list_unique_order(listt, string):
    """ Given a list *listt* we want to append it with a string
        We do so by avoiding repetitions, and making the list ordered
        No need to be super efficient here
    """
    if string not in listt:
        listt.append(string)
    listt.sort()
    return listt

class Dict2D(dict):
    """ Basically encodes what will be a "2D dict". Each coefficient of this 2D dict 
        will correspond to a specific couple (xkey, ykey)
        Note that this 2D structure is somehow sparse : not all couplings are possible

        Dict2D is itself a dict of dict 
            Its contents will be stored in Dict2D[xkey][ykey]
            but we avoid touching this directly because when defining a coefficient we want 
            to maintain a list of all the existing (xkey, ykey)
        We set the coefficients of Dict2D via
            Dict2D.setvalue(xkey, ykey, value)
            Don't want to overwrite the existing set function
        We read the coefficients of Dict2D via
            Dict2D.getvalue(xkey, ykey, value)
            Don't want to overwrite the .get method of dicts

        Dict2D.xkeys() is a list of all the xkeys 
        Dict2D.ykeys() is a list of all the ykeys
            Both will be useful for iterating over the 2D dict
            Both are updated when using .setvalue
    """
    def __init__(self):
        super().__init__() # self is a dict
        self.xkeys_value = []
        self.ykeys_value = []
    
    def xkeys(self): # to mimic dict.keys()
        return self.xkeys_value
    
    def ykeys(self): # same
        return self.ykeys_value
    
    def isdefined(self, xkey, ykey):
        # do we print something here?
        if xkey in self.keys():
            if ykey in self[xkey].keys():
                return True
        return False

    def setvalue(self, xkey, ykey, value):
        if xkey in self.keys():
            self[xkey] = { **self[xkey], ykey : value }
        else:
            self[xkey] = {ykey : value}
        self.xkeys_append(xkey)
        self.ykeys_append(ykey)
    
    def getvalue(self, xkey, ykey):
        if self.isdefined(xkey, ykey):
            return self[xkey][ykey]
    
    def delvalue(self, xkey, ykey):
        if self.isdefined(xkey, ykey):
            del self[xkey][ykey]
            # is there another ykey somewhere?
            if any([ self.isdefined(key, ykey) for key in self.xkeys() ]): # true if all false
                pass
            else:
                self.ykeys_value.remove(ykey)
            # now we look at xkey
            if self[xkey] == {}:
                del self[xkey]
                self.xkeys_value.remove(xkey)
        
    def xkeys_append(self, value):
        self.xkeys_value = update_list_unique_order(self.xkeys_value, value)
        
    def ykeys_append(self, value):
        self.ykeys_value = update_list_unique_order(self.ykeys_value, value)
    
    def getvalue_x(self, xkey):
        return self[xkey]
    
    def getvalue_y(self, ykey):
        return { xkey : self[xkey][ykey] for xkey in self.xkeys() if ykey in self[xkey].keys() }


class Results(Dict2D):
    """ A "2D dict" for which each coefficient is a Records, 
        corresponding to a specific couple (record_name, solver_name)

        Then we have a couple of methods for plotting, saving the results
    """
    def __init__(self, problem):
        super().__init__()
        self.param = Parameters(problem.data)

    def set_records(self, solver_name, records_dict):
        for record_name in records_dict.keys():
            self.setvalue(solver_name, record_name, records_dict[record_name])

    def save(self):
        for solver_name in self.xkeys():
            for record_name in self.ykeys():
                record = self[solver_name][record_name]
                record.save() # see Record.save

    def extract_list_given_solver(self, solver_name):
        dico = self.getvalue_x(solver_name)
        return [record for record in dico.values()]

    def extract_list_given_record(self, record_name):
        dico = self.getvalue_y(record_name)
        return [record for record in dico.values()]
    
    def process_values(self, xaxis_time = False):
        # process the values (extracts mean, min, max), make it ready for analysis
        # provide each record with an alternate .xaxis_time to plot wrt time
        for record_name in self.ykeys(): # all the kinds of records
            for solver_name in self.xkeys(): # all the solvers
                if self.isdefined(solver_name, record_name): # if this solver has this record
                    record = self.getvalue(solver_name, record_name) # we now have a Record
                    if xaxis_time: # there should be a "time_epoch" Records for every solver
                        time_value_repetition = self[solver_name]["time_epoch"].value_repetition
                        record.process_values(time_value_repetition)
                    else:
                        record.process_values()
    
    def load(self, solvers_to_load):
        # solvers_to_load is a list of strings referring to Solver.name
        # first get a list of all possible records (we omit the temp records starting with '_')
        # note that pointer.__name__ is the name we use in class BLAH(Record)
        # while pointer.name is a string we define as a class variable before init, which we use to save
        list_record_name = [ pointer.name for pointer in Records.__subclasses__() if pointer.__name__[0] != '_' ]
        for solver_name in solvers_to_load:
            for record_name in list_record_name:
                    # we check if the couple (solver_name, record_name) was saved
                    # its path would be:
                    path = os.path.join(self.param.output_folder, solver_name + '-' + record_name)
                    # if it is there, we load it
                    if os.path.exists(path):
                        with open(path, 'rb') as fp:
                            record = pickle.load(fp)
                            self.setvalue(solver_name, record_name, record)
                        print(f"Loaded record for the solver {solver_name} : {record_name}")
        return

    def plot_parallel(self, record_name, xaxis_time = False):
        """ record_name is the name of some Records() 
            We assume that our Result contains that Records() for one or many Solver()

            We are going to plot the curves of this quantity for each solver.
            The quantity to plot is in Records.value_repetition 
            The parameters of the plot are in Records.plot
            See Plot_param() for the default values.
        """
        # Get a list of Records() corresponding to all the Solvers() recording record_name
        record_list = self.extract_list_given_record(record_name)
        # first setup of the figure
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{underscore}')
        plt.rc('font', family='sans-serif')
        param_plot = record_list[0].param.plot # all these Records should have the same figsize/dpi parameters
        plt.figure(figsize=param_plot.figsize, dpi=param_plot.dpi)
        # loop over the algorithms (each will have different marker)
        markers = ["^-", "d-", "*-", ">-", "+-", "o-", "v-", "<-"]
        for record, marker in zip(record_list, markers):
            param_plot = record.param.plot # each of those have different legend name
            # plot
            if xaxis_time:
                xaxis = record.xaxis_time
                param_plot.xlabel = "Time (s)"
            else:
                xaxis = record.xaxis
            plt.semilogy(xaxis, record.value_avg, marker, label=record.solver_name, lw=2)
            if param_plot.show_variance:
                plt.fill_between(xaxis, record.value_min, record.value_max, alpha=0.2)

        # All curves are plotted. Make it look good.
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=30)
        if param_plot.xlabel != "":
            plt.xlabel(param_plot.xlabel, fontsize=25)
        if param_plot.ylabel != "":
            plt.ylabel(param_plot.ylabel, fontsize=25)
        plt.grid(True)
        if param_plot.title is not None:
            plt.title(param_plot.title, fontsize=25)
        if param_plot.do_we_save:
            if not os.path.exists(param_plot.fig_folder):
                os.makedirs(param_plot.fig_folder)
            if xaxis_time:
                fig_path = param_plot.fig_path + '-time'
            else:
                fig_path = param_plot.fig_path
            plt.savefig(fig_path+'.pdf', bbox_inches='tight', pad_inches=0.01)
        else: # no save, just plot it
            plt.show()
        return

    def plot_all(self, xaxis_time=False):
        for record_name in self.ykeys():
            if record_name != "time_epoch": # we don't want to plot that. Could use a blacklist if need to generalize
                print(f"Start plotting the results for {record_name}")
                self.plot_parallel(record_name)
                if xaxis_time: # second pass for the plots wrt time
                    self.plot_parallel(record_name, xaxis_time = True) 


