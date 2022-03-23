import os
import config


##################################################
class Parameters():
    # parameters for the algorithms, the plots etc
    def __init__(self, data):
        # parameters for the solvers
        self.lr = config.lr # learning rate / stepsize
        self.nb_epoch = config.epochs
        self.tol = config.tol
        self.n_repetition = config.n_repetition
        self.initialization = config.initialization # initial point for all algorithms
        # Options for saving, logging, plotting
        self.solvers_to_run = config.solvers_to_run
        self.solvers_to_load = config.solvers_to_load
        self.verbose = config.verbose
        self.measure_time = config.measure_time
        self.do_we_save=config.save,
        self.data_name = data.name
        self.data_path = data.path
        self.output_folder = os.path.join(config.output_path, self.data_name)
        self.plot = Plot_param(self)

class Plot_param(): 
    # A class to encode parameters to display plots. 
    def __init__(self, parameters): 
        self.record_name = "" # will be set for each Records !!! copy issues?
        self.data_name = parameters.data_name
        self.x = None # options to display records as a plot
        self.y = None
        self.xlabel = ""
        self.ylabel = ""
        self.title = f"{self.data_name}" # default
        self.dpi = config.dpi
        self.figsize = (9, 6)
        self.fig_folder = parameters.output_folder # the folder output/data/
        self.fig_name = None # for the plots in pdf
        self.fig_path = None # complete path for the pdf plot
        self.measure_time = parameters.measure_time
        self.threshold = config.plot_threshold # option to alter the plot
        self.show_variance = True # default
        self.do_we_save = parameters.do_we_save
    
    def set_record_name(self, name):
        # Plot_param() is init at the beggining of the script
        # the record name is given when Records() is init
        # and we need to propagate it to the .fig_path
        self.record_name = name
        self.fig_name = f"{self.data_name}-{self.record_name}" # for the plots in pdf
        self.fig_path = os.path.join(self.fig_folder, self.fig_name) # complete path for the pdf plot

##################################################