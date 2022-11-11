import os
import time
import yaml

def get_config():
    # gets the config 
    path = 'config.yml'
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream) # we get a dict from the file
    config = merge_default_param(config) # apply the hardwritten default parameters if needed
    # does some cleaning on the solvers parameters
    config = apply_default_solver_parameters(config) # decide which parameter to apply per solver
    config['solvers_parameters']['solvers_to_run'] = get_list_solver_to_run(config)
    config['solvers_parameters']['solvers_to_load'] = get_list_solver_to_load(config)
    config['results']['records_to_record'] = get_list_record_to_record(config)
    return config

def merge_default_param(config):
    # local user > local default > global user > global default 
    # merge default and user values at a global level
    with open('src/config_default.yml', 'r') as stream:
        config_default = yaml.safe_load(stream)
    config['problem'] = { **config_default['problem'] , **config['problem'] } 
    config['results'] = { **config_default['results'] , **config['results'] } 
    config['solvers_parameters'] = { **config_default['solvers_parameters'] , **config['solvers_parameters'] } 
    # now merge global parameters and per-solver parameters
    solvers_list = config['solvers']
    global_param = config['solvers_parameters']
    for idx, list_item in enumerate(solvers_list):
        if isinstance(list_item, str):
            name_solver = list_item
            local_param = {}
        if isinstance(list_item, dict):
            name_solver = list(list_item.keys())[0] # there should be only one key
            local_param = list_item[name_solver]
        param = { **global_param, 'flavor_name' : name_solver }
        # todo : deal with local default parameters
        solvers_list[idx] = { name_solver : { **param, **local_param } }
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

def get_list_record_to_record(config):
    list_of_records = config['results']['records_to_plot']
    if 'records_to_compare' in config['results'].keys():
        for comparison in config['results']['records_to_compare']:
            for record_name in comparison:
                list_of_records.append(record_name)
    if config['results']['measure_time']: # special
        list_of_records.append("time_epoch")
    return list(set(list_of_records)) # remove duplicates



##################################################
class Parameters():
    # parameters for the algorithms, the plots etc
    def __init__(self, data):
        # loads the config file into a dict
        config = get_config()
        self.config = config # just in case
        # parameters for the solvers IS THIS NECESSARY?
        self.nb_epoch = config['solvers_parameters']['nb_epochs']
        self.tol = float(config['solvers_parameters']['tolerance'])
        self.n_repetition = config['solvers_parameters']['nb_repetition']
        self.distribution = config['solvers_parameters']['distribution']
        self.initialization = config['solvers_parameters']['initialization'] # initial point for all algorithms
        self.solvers = config['solvers'] # list of dict containing all we need to run each algorithm
        self.solvers_to_run = config['solvers_parameters']['solvers_to_run']
        self.solvers_to_load = config['solvers_parameters']['solvers_to_load']
        self.records_to_record = config['results']['records_to_record']
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
        self.figsize = (18, 12)
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