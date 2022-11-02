import os
import time
import yaml

def get_config():
    # gets the config 
    path = 'config.yml'
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    # merge default and user values
    with open('src/config_default.yml', 'r') as stream:
        config_default = yaml.safe_load(stream)
    config = { **config_default , **config } 
    # does some cleaning on the solvers parameters
    config = apply_default_solver_parameters(config)
    config['solvers_parameters']['solvers_to_run'] = get_list_solver_to_run(config)
    config['solvers_parameters']['solvers_to_load'] = get_list_solver_to_load(config)
    return config
        
def apply_default_solver_parameters(config):
    # input: a list of strings or dict of the form { name_solver : dict_param }
    # output: 
    # - replace each string by a dict of the form { string : {} }
    # - complement each solver parameters with the default ones
    # - add the flavor_name which by default is the solver name
    lst = config['solvers']
    dict_default_param = config['solvers_parameters']
    for idx, solver in enumerate(lst):
        if isinstance(solver, str):
            lst[idx] = { solver : { **dict_default_param, 'flavor_name' : solver } }
        if isinstance(solver, dict):
            name_solver = list(solver.keys())[0] # there should be only one key
            lst[idx] = { name_solver : { **dict_default_param, 'flavor_name' : name_solver, **solver[name_solver] } }
    return config

def get_list_solver_to_run(config):
    return [key for solver in config['solvers'] for key in solver.keys() if 'load' not in solver[key] or not solver[key]['load']]

def get_list_solver_to_load(config):
    return [key for solver in config['solvers'] for key in solver.keys() if 'load' in solver[key] and solver[key]['load']]


##################################################
class Parameters():
    # parameters for the algorithms, the plots etc
    def __init__(self, data):
        # loads the config file into a dict
        config = get_config()
        self.config = config # just in case
        # parameters for the solvers
        self.lr = config['solvers_parameters']['lr'] # learning rate / stepsize
        self.nb_epoch = config['solvers_parameters']['nb_epochs']
        self.tol = float(config['solvers_parameters']['tolerance'])
        self.n_repetition = config['solvers_parameters']['nb_repetition']
        self.distribution = config['solvers_parameters']['distribution']
        self.initialization = config['solvers_parameters']['initialization'] # initial point for all algorithms
        self.solvers = config['solvers'] # list of dict containing all we need to run each algorithm
        self.solvers_to_run = config['solvers_parameters']['solvers_to_run']
        self.solvers_to_load = config['solvers_parameters']['solvers_to_load']
        # Options for saving, logging, plotting
        self.measure_time = config['results']['measure_time']
        self.records_to_plot = config['results']['records_to_plot']
        self.do_we_plot = config['results']['do_we_plot'] or config['results']['save_plot']
        self.save_data = config['results']['save_data']
        self.save_plot = config['results']['save_plot']
        self.do_we_save = self.save_data or self.save_plot
        self.log_file = config['results']['log_file']
        self.measure_time = config['results']['measure_time']
        self.data_name = data.name
        self.data_path = data.path
        self.xp_name = config['results']['xp_name']
        self.output_folder = os.path.join(config['results']['output_path'], config['results']['xp_name'], self.data_name)
        # make sure that the folder exists everytime Parameters.output_folder is used
        if self.do_we_save:
            if not os.path.exists(self.output_folder): 
                os.makedirs(self.output_folder)
            # save the config into a .yml file so it can be reused in the future
            with open(os.path.join(config['results']['output_path'], config['results']['xp_name'], 'config.yml'), 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)
        self.plot = Plot_param(self) # this is last bc it calls Parameters
    
    


class Plot_param(): 
    # A class to encode parameters to display plots. 
    def __init__(self, parameters): 
        config = get_config()
        self.record_name = "" # will be set for each Records !!! copy issues?
        self.data_name = parameters.data_name
        self.x = None # options to display records as a plot
        self.y = None
        self.xlabel = ""
        self.ylabel = ""
        self.title = f"{self.data_name}" # default
        self.dpi = config['results']['dpi']
        self.figsize = (9, 6)
        self.fig_folder = parameters.output_folder # the folder output/data/
        self.fig_name = None # for the plots in pdf
        self.fig_path = None # complete path for the pdf plot
        self.measure_time = parameters.measure_time
        self.threshold = float(config['results']['plot_threshold']) # option to alter the plot
        self.show_variance = True # default
        self.do_we_plot = parameters.do_we_plot
        self.save_data = parameters.save_data
        self.save_plot = parameters.save_plot
        self.do_we_save = parameters.do_we_save
    
    def set_record_name(self, name):
        # Plot_param() is init at the beggining of the script
        # the record name is given when Records() is init
        # and we need to propagate it to the .fig_path
        self.record_name = name
        self.fig_name = f"{self.data_name}-{self.record_name}" # for the plots in pdf
        self.fig_path = os.path.join(self.fig_folder, self.fig_name) # complete path for the pdf plot

##################################################