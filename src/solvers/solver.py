from abc import ABCMeta, abstractmethod
import numpy as np
import time
import logging

from src.parameters import Parameters
from src.solvers import Records



""" The class Solver() can be declined in many subclasses, each defining a specific solver
    Each solver can be different from the others, but here they share:

    - Storing some essential information such as the Problem(), the Parameters()
    - Running one epoch via .run_epoch (it is used later in run and run_repetition)
    - Recording some value of iterest to be plotted later
        Those values are stored in Solver.records which is a dict of Records
        See the class Records() for more details
"""

class Solver(metaclass=ABCMeta):

    def __init__(self, problem):
        if not hasattr(self, 'name'): # sometimes the name is already defined
            self.name = ""
        self.problem = problem
        self.loss = problem.loss
        self.regularizer = problem.regularizer
        self.param = Parameters(problem.data) # can create problem with *reload*
        self.x = None
        self.epoch_running_time = 0.0
        self.total_running_time = 0.0
        self.records = {} # { record_name : Record() }
        if self.param.measure_time:
            self.append_records("time_epoch") 
    """
    def append_records(self, *tuple_of_class):
        # tag the Records with the name of the Solver they'll belong to
        for cls in tuple_of_class:
            cls.solver_name = self.name
            cls.data_name = self.problem.data.name
        update = { cls.name : cls for cls in tuple_of_class }
        self.records = { **self.records, **update }
    """
    def append_records(self, *tuple_of_class_name):
        # all the existing records:
        list_records = [ record for record in Records.__subclasses__() if record.__name__[0] != '_' ]
        dict_of_records = {}
        # tag the Records with the name of the Solver they'll belong to
        for cls in list_records:
            if cls.name in tuple_of_class_name: # we want this Record for this Solver
                record_instance = cls(self)
                record_instance.solver_name = self.name
                record_instance.data_name = self.problem.data.name
                dict_of_records = { **dict_of_records, cls.name : record_instance }
        self.records = { **self.records, **dict_of_records }

    # Runs the Solver. It calls a handful of subroutines, defined after
    def run(self):
        # startup
        self.initialization_variables() # put all states at "zero"
        self.store_records() # updates self.records[all].value at iteration 0
        # main loop
        for cnt in range(1, self.param.nb_epoch + 1):
            # one epoch
            if self.param.measure_time:
                start_time = time.time()
                self.run_epoch()
                self.epoch_running_time = time.time() - start_time
                self.total_running_time += self.epoch_running_time
            else: # we just run the epoch
                self.run_epoch()
            # storing, verbosing, and stop-or-keep-looping
            self.store_records()
            self.log_information(cnt)
            if self.we_have_to_break():
                break # out of the for loop
        return

    # Some of the blocks needed to .run() the Solver
    def initialization_variables(self):
        # standard, can be overwritten if needed in child Solver
        # the initial iterate
        init = self.param.initialization
        if isinstance(init, str):
            if init == "zeros":
                self.x = np.zeros(self.problem.dim)
            # we could do "random" as well but need to be careful with seed
        else: # we assume it is a np.array
            self.x = init
        # the measure of time
        self.epoch_running_time = 0.0
        self.total_running_time = 0.0

    def store_records(self): 
        # All the work is done in Records().store()
        for record in self.records.values():
            record.store(self)
        
    @abstractmethod
    def run_epoch(self):
        pass # This is to be defined for each child Solver

    def log_information(self, cnt):
        text = f"| Solver : {self.name} "
        text += f"| End of epoch #{cnt:d} "
        if self.param.measure_time:
            text += f"| ran in {self.epoch_running_time:f}s "
        for record in self.records.values(): 
            if record.name != "time_epoch": # to avoid doublons
                text += f"| {record.name} {record.value[-1]:f} "
        text += "|"
        logging.info(text)
    
    def we_have_to_break(self):
        return self.param.tol is not None and self.records["gradient_norm"].value[-1] <= self.param.tol

    # An upper function running a solver multiple times
    def run_repetition(self):
        np.random.seed(0)  # random seed to reproduce the experiments
        logging.info(f"------START {self.name}------")
        for i in range(self.param.n_repetition):
            logging.info(f"{i+1}-th repetition:")
            self.run() # one run of the solver, see above
            self.archive_records_repetition() # archive records and prepare for an another run
        logging.info(f"------END {self.name}------")
        return

    def archive_records_repetition(self): 
        for record in self.records.values(): # record is a Records() class
            record.value_repetition.append(record.value) # carved in stone
            record.value = [] # run is over so we can wipe it
        return

    # Other needed methods
    def save_records(self):
        # save the list of lists contained in each Records.value_repetition
        # or maybe self.records itself?? TBC
        for record in self.records.values():
            record.save()
    
