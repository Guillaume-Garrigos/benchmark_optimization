import os
import yaml
import copy
import itertools
import shutil
import numpy as np

def secure_config(path_to_config=None):
    # keeps a copy of the config file with a fixed location/name
    # so we can access it anytime
    if path_to_config is None:
        # we assume that the path was 'config.py'
        path = 'config.yml'
    else:
        path = path_to_config
    # secure the config file so we can retrieve it later
    shutil.copy(os.path.join('', path), os.path.join('src','config.yml'))
    return

def release_config():
    # deletes the copy of the config file
    os.remove(os.path.join('src','config.yml'))
    return

def get_config():
    # gets the config 
    path = os.path.join('src','config.yml')
    with open(path, 'r') as file:
        config = yaml.safe_load(file) # we get a dict from the file
    # apply the hardwritten default parameters if needed
    config = merge_default_param(config) 
    # does some cleaning on the solvers parameters
    # decide which parameter to apply per solver
    config = apply_default_solver_parameters(config) 
    # if we want to do grid search on some parameters, duplicate the solvers as needed
    config = deal_with_grid_search(config)
    # read the parameters and extract some useful information
    config['solvers_parameters']['solvers_to_run'] = get_list_solver_to_run(config)
    config['solvers_parameters']['solvers_to_load'] = get_list_solver_to_load(config)
    config['results']['records_to_plot'] = get_list_record_to_plot(config)
    config['results']['records_to_record'] = get_list_record_to_record(config)
    return config


def merge_default_param(config):
    # local user > local default > global user > global default 
    # merge default and user values at a global level
    with open('src/config_default.yml', 'r') as file:
        config_default = yaml.safe_load(file)
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


def deal_with_grid_search(config):
    # if any solver gets called with a 'grid_search' parameter
    # create as much solver calls as needed, each with its own parameter
    # do what is needed so that this grid search can be plotted later
    # we assume that in config['solvers'] there is only dict and no string since this is dealt with in apply_default_solver_parameters

    # In this we'll gather all the grid search info we'll need
    grid_search_info = []
    # this is going to replace config['solvers'] at the end
    new_config_solvers = [] 
    # we explore what is in config['solvers']
    for solver_dict in config['solvers']:
        solver_name = list(solver_dict)[0]
        param_dict = solver_dict[solver_name]
        if 'grid_search' in param_dict.keys():
            # we have work to do
            grid_search_param_names = list(param_dict['grid_search'].keys())
            grid_search_info = merge_list(grid_search_info, grid_search_param_names)
            # if the values come as a dict, replace it with a list
            for parameter_name in grid_search_param_names:
                param_dict['grid_search'][parameter_name] = set_param_grid(param_dict['grid_search'][parameter_name])
            # loop over all possible parameter combinations
            list_list_values = [ dico['grid'] for dico in param_dict['grid_search'].values() ] #[(1.0, 2),(1.1, 2),.., (2.0, 5)]
            unique_indexes = [ list(range(len(liste))) for liste in list_list_values ] #[(0, 0),(1, 0),.., (10, 4)]
            for param_tuple, index in zip(itertools.product(*list_list_values), itertools.product(*unique_indexes)):
                # we want to create in config['solvers'] a new entry with those parameters
                # first copy all the other parameters
                new_entry_param = copy.deepcopy(param_dict)
                # leave a trace of what we did could be useful later? TODO
                new_entry_param['grid_search_param_names'] = grid_search_param_names 
                new_entry_param['grid_search_result'] = True
                # set a unique name to the flavor
                new_entry_param['flavor_name'] += '_'+'_'.join([str(index[i]) for i in range(len(index))])
                # set the specific parameters for this entry
                for param_name, param_value in zip(grid_search_param_names, param_tuple):
                    new_entry_param[param_name] = param_value
                # the dict is ready, we can add it to the config dict
                new_config_solvers.append({solver_name : new_entry_param})
        else:
            # we have a solver with no grid search. So we will want to keep it
            new_config_solvers.append(solver_dict)
    # we now update config['solvers']. Note that by doing so, we 
    # 1) kept all solvers with no grid search
    # 2) got rid of all solvers with grid search
    # 3) created a bunch of copies of these solvers with appropriate parameters
    config['solvers'] = new_config_solvers
    # we store the list of the names of the parameters we want to grid-loop on
    config['results']['grid_search_to_plot'] = grid_search_info
    return config

def set_param_grid(input):
    # Input: a dictionnary with some parameters, or a list
    # Output: a dictionnary with at least the list of parameters
    # Generated with range(parameters) for instance or more complicated
    if isinstance(input, list): # we keep it simple
        return { 'grid' : input , 'scale' : 'linear', 'number' : len(input) }
    if isinstance(input, dict): # we have some work to do
        input['scale'] = input.get('scale', 'linear') # default is linear
        input['number'] = input.get('number', 10) # default is 10
        if input['scale'] == 'linear':
            input['grid'] = np.linspace(input['min'], input['max'], num=input['number'])
            return input
        elif input['scale'] == 'log':
            input['grid'] = np.geomspace(input['min'], input['max'], num=input['number'])
            return input

def get_list_solver_to_run(config):
    return [key for solver in config['solvers'] for key in solver.keys() if 'load' not in solver[key] or not solver[key]['load']]

def get_list_solver_to_load(config):
    return [key for solver in config['solvers'] for key in solver.keys() if 'load' in solver[key] and solver[key]['load']]

def get_list_record_to_plot(config):
    if 'records_to_plot' in config['results'].keys() and config['results']['records_to_plot'] is not None:
        return config['results']['records_to_plot']
    else:
        return []

def get_list_record_to_record(config):
    list_of_records = copy.deepcopy(config['results']['records_to_plot'])
    if 'records_to_compare' in config['results'].keys():
        for comparison in config['results']['records_to_compare']:
            for record_name in comparison:
                list_of_records.append(record_name)
    else: # we set it to emptylist so we are not annoyed anymore later
        config['results']['records_to_compare'] = []
    if config['results']['measure_time']: # special
        list_of_records.append("time_epoch")
    return list(set(list_of_records)) # remove duplicates

def merge_list(list1, list2):
    # https://stackoverflow.com/a/71861638
    return [*dict.fromkeys(list1+list2)]

##################################################
class Parameters():
    # parameters for the algorithms, the plots etc
    def __init__(self, data):
        # loads the config file into a dict
        config = get_config()
        # everything which is below is a mess, I shouldn't copy values like that one by one beacause everytime I add a parameter it becomes endless
        # so better maybe keep the structure of the config file?
        self.config = config
        self.problem = config['problem']
        self.results = config['results']
        self.solvers_parameters = config['solvers_parameters']

        # parameters for the solvers IS THIS NECESSARY?
        self.nb_epoch = config['solvers_parameters']['nb_epochs']
        self.tol = float(config['solvers_parameters']['tolerance'])
        self.n_repetition = config['solvers_parameters']['nb_repetition']
        self.distribution = config['solvers_parameters']['distribution']
        self.initialization = config['solvers_parameters']['initialization'] # initial point for all algorithms
        self.solvers = config['solvers'] # list of dict containing all we need to run each algorithm
        self.solvers_to_run = config['solvers_parameters']['solvers_to_run']
        self.solvers_to_load = config['solvers_parameters']['solvers_to_load']
        self.records_to_plot = config['results']['records_to_plot']
        self.records_to_record = config['results']['records_to_record']
        # Options for saving, logging, plotting
        self.measure_time = config['results']['measure_time']
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
        self.threshold = float(config['results']['plot_threshold_down']) # option to alter the plot
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