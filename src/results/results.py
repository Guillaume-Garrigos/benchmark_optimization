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
            Dict2D.getvalue(xkey, ykey)
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
                if self.isdefined(solver_name, record_name):
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

    def do_we_plot_curves(self, record_names):
        # we are tasked to plot the curves corresponding to a list of records (usually one record)
        # and to possibly various solvers
        # we don't want to plot it if all the solvers are part of a grid search
        # unless this default behavior is overriden in the config file
        for record_name in record_names:
            record_list = self.extract_list_given_record(record_name)
            for record in record_list:
                if (not record.param.config.get('grid_search_result')) or (self.param.results['grid_search_curves_plot']):
                    # ok we found one solver which is not part of the grid search 
                    # so it is worth creating a plot
                    return True
        # we finish our search, there is nothing but gridsearch so we don't plot
        return False

    def plot_parallel(self, record_names, xaxis_time = False):
        """ record_name is the name of some Records(), eventually a list of names
            We assume that our Result contains that Records() for one or many Solver()

            We are going to plot the curves of this quantity for each solver.
            The quantity to plot is in Records.value_repetition 
            The parameters of the plot are in Records.plot
            See Plot_param() for the default values.

            We likely won't plot records if they are part of a grid search
        """
        if isinstance(record_names, str): # classic case in which one plot = one record
            record_names = [record_names]
        if not self.do_we_plot_curves(record_names): # if not we just do nothing
            return
        else: # we plot
            # set parameters
            param_plot = self.param.plot
            # first setup of the figure TODO check if this isn't slowing down everything, put a if maybe
            if self.param.results['latex']:
                plt.rc('text', usetex=True)
                plt.rc('text.latex', preamble=r'\usepackage{underscore}')
            plt.rc('font', family='sans-serif')
            # create the canvas
            plt.figure(figsize=param_plot.figsize, dpi=param_plot.dpi)

            # loop over possible records to plot
            there_is_mixed_records = len(record_names) > 1
            for record_name in record_names:
                # Get a list of Records() corresponding to all the Solvers() recording record_name
                record_list = self.extract_list_given_record(record_name)
                # add curves to the canvas
                label_append = ""
                if there_is_mixed_records:
                    param_plot.xlabel = record_list[0].param.plot.xlabel
                    param_plot.ylabel = "" # we don't want to mix the ylabels for different records
                    label_append += ':' + record_name # instead we display it in the legend
                self.plot_curves_given_record(record_list, xaxis_time=xaxis_time, label_append=label_append)

            # All curves are plotted. Make it look good.
            plt.tick_params(labelsize=20)
            plt.legend(fontsize=30, loc='best')
            if param_plot.title is not None:
                plt.title(param_plot.title, fontsize=25)
            plt.grid(True)
            # set the labels for axis
            if xaxis_time:
                xlabel = "Time (s)"
            else:
                xlabel = "Effective Passes"
            if there_is_mixed_records: # we don't display yaxis
                ylabel = "" 
            else: # get the Record from the unique record name and extract the ylabel
                ylabel = self.extract_list_given_record(record_names[0])[0].param.plot.ylabel
            plt.xlabel(xlabel, fontsize=25)
            plt.ylabel(ylabel, fontsize=25)
            
            # finally, show/save the figures
            if param_plot.save_plot:
                fig_name = "_vs_".join(record_names)
                fig_path = os.path.join(param_plot.fig_folder, fig_name)
                if xaxis_time:
                    fig_path = fig_path + '-time'
                plt.savefig(fig_path+'.pdf', bbox_inches='tight', pad_inches=0.01)
            if param_plot.do_we_plot:
                plt.show()
            return

    def plot_curves_given_record(self, record_list, **args):
        # loop over the algorithms (each will have different marker)
        markers = ["^-", "d-", "*-", ">-", "+-", "o-", "v-", "<-"]
        values_are_negative = any([any(record.value_avg<=0.0) for record in record_list])
        for record, marker in zip(record_list, markers):
            # we won't plot the record if its part of a grid search unless specified
            if (not record.param.config.get('grid_search_result')) or (self.param.results['grid_search_curves_plot']):
                param_plot = record.param.plot # each of those have different legend name
                # set the xaxis
                if args['xaxis_time']:
                    xaxis = record.xaxis_time
                    param_plot.xlabel = "Time (s)"
                else:
                    xaxis = record.xaxis
                # if the values are >0 we plot with logscale. Otherwise, normal scale
                label = record.flavor_name + args['label_append']
                # now we plot
                if values_are_negative:
                    plt.plot(xaxis, record.value_avg, marker, label=label, lw=2)   
                else:             
                    plt.semilogy(xaxis, record.value_avg, marker, label=label, lw=2)
                # eventually show variance (for repetitions)
                if param_plot.show_variance:
                    plt.fill_between(xaxis, record.value_min, record.value_max, alpha=0.2)

    def plot_all(self, xaxis_time=False):
        # 1) plot the curves for each records
        for record_name in self.param.records_to_plot:
            # here we simply plot this record, for every solver it is related to
            if record_name in self.ykeys(): # safecheck
                print(f"Start plotting the results for {record_name}")
                self.plot_parallel(record_name)
                if xaxis_time: # second pass for the plots wrt time
                    self.plot_parallel(record_name, xaxis_time = True)
        # 2) plot curves when comparing records with each other
        for comparison in self.param.config['results']['records_to_compare']:
            # comparison is a list of Record names
            # we'll now plot different records on the same plot, for every solver
            if all( record_name in self.ykeys() for record_name in comparison ): # safecheck
                print(f"Start plotting the compared results for: {comparison}")
                self.plot_parallel(comparison)
        # 3) plot the results on the grid search
        for parameter_name in self.param.config['results']['grid_search_to_plot']:
            print(f"Start plotting the grid search results for the parameter: {parameter_name}")
            self.plot_grid_search(parameter_name)
    
    def plot_grid_search(self, parameter_name):
        # We did a grid search on parameter_name for one or many solvers
        # we want to plot those results
        config = self.param.config
        if not config['results']['grid_search_comparison_plot']: # we dont plot
            return
        else: # lets go
            # set parameters
            param_plot = self.param.plot
            # first setup of the figure TODO check if this isn't slowing down everything, put a if maybe
            if self.param.results['latex']:
                plt.rc('text', usetex=True)
                plt.rc('text.latex', preamble=r'\usepackage{underscore}')
            plt.rc('font', family='sans-serif')
            # create the canvas
            plt.figure(figsize=param_plot.figsize, dpi=param_plot.dpi)
            # Loop over all solvers names
            solver_names = [*dict.fromkeys(config['solvers_parameters']['solvers_to_run'])]
            for solver_name in solver_names:
                # we're gonna put in those lists all the points we want to plot for this solver_name
                scatter_parameters = [] 
                scatter_records = [] 
                scatter_error_lower, scatter_error_upper = [],[] #  for the error below and above
                xscale = 'linear' # default value for the plot
                # we collect all the instances which are part of this grid search
                for solver_instance in config['solvers']:
                    if solver_name in solver_instance.keys():
                        # Check if that instance is part of the grid search
                        instance_param = solver_instance[solver_name]
                        if instance_param.get('grid_search') and parameter_name in instance_param['grid_search'].keys(): # ok we want to add it to our graph
                            # we access the Records of this instance
                            flavor_name = instance_param['flavor_name']
                            record_name = instance_param['grid_search'][parameter_name].get('record', 'function_value')
                            record = self.getvalue(flavor_name, record_name)
                            # now we dig into it and extract the desired values
                            scatter_parameters.append(instance_param[parameter_name]) # the value of the parameter for this instance
                            scatter_records.append(record.value_avg[-1]) # the last value for this record
                            scatter_error_lower.append(record.value_avg[-1] - record.value_min[-1])
                            scatter_error_upper.append(record.value_max[-1] - record.value_avg[-1])
                            # check a few display parameters
                            if instance_param['grid_search'][parameter_name]['scale'] == 'log':
                                xscale = 'log' # if only one method wants logscale we do it
                # ok we have all the data about this solver. We can plot it.
                plt.errorbar(scatter_parameters, scatter_records, yerr=[scatter_error_lower, scatter_error_upper], fmt='x') # https://stackoverflow.com/a/43990689
                plt.xscale(xscale)
                plt.yscale('log')



                # extract what we want to plot
            # check if there is something to plot? and repetitions?
            # add to the graph a scatter plot. Legend is the solver name
        # Done
        

