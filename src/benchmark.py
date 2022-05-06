import os
import logging
import time

#from . import config
import config as config
from src.problem.data import Data
from src.problem.problem import Problem
from src.results.results import Results
from src.solvers.solver import Solver # this does implicit things in its __init__.py

def run_solvers(problem):
    """ given a Problem(), run multiple solvers against it, and return Results() """
    # setup the logging and outputing
    data_name = problem.data.name
    folder_path = os.path.join(config.output_path, data_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    logging.basicConfig(
        filename=os.path.join(folder_path, config.log_file),
        level=logging.INFO, format='%(message)s')
    logging.info(time.ctime(time.time()))
    # Run solvers
    print(f"Running each solver on {data_name}")
    results = Results(problem) # a 2D dict, each coefficient will be a Record
    # Get a list of the available solvers. Nasty but at least we do not need to come back here if create a new solver.
    list_solvers = [ algo for algo in Solver.__subclasses__() if algo.__name__[0] != '_' ]
    # !!! to be removed
    #print("List of available solvers : "+str([algo.name for algo in list_solvers]))
    # now we run each solver and stack the results
    for algo in list_solvers:
        algo_name = algo.name # accessible as a class variable (even before init)
        if algo_name in problem.param.solvers_to_run : # we want to run it
            print(f"Running solver '{algo_name}' on {data_name}")
            solver = algo(problem) # Instanciate a solver
            solver.run_repetition() # run the solver with repetitions
            results.set_records(solver.name, solver.records) # stores all the records from solver.records
    # process the values (extracts mean, min, max), make it ready for analysis
    # and provide each record with an alternate .xaxis_time to plot wrt time
    print("Process the results")
    results.process_values(config.measure_time)
    # eventually saves each records as a file into the folder defined in config
    results.save()
    logging.shutdown() # otherwise python keeps its hands on the log .txt file
    return results


def benchmark_plot(results):
    # We eventually complete results with records to load
    # we assume each record is loaded with a following path:
    #   output_directory/data_name/solver_name-record_name
    results.load(config.solvers_to_load)
    results.plot_all(xaxis_time = config.measure_time)
    return

def benchmark_datasets():
    # loop over all datasets
    for data_name, data_path in zip(config.dataset_names, config.dataset_paths):
        print(f"START running {data_name}")
        # setup the main components of the problem, depending on the dataset
        data = Data()
        data.read(data_name, data_path) # gathers the components of the data
        problem = Problem(data) # gathers the components defining the problem
        # solve the problem with multiple solvers
        results = run_solvers(problem)
        # plot the results
        if config.verbose:
            benchmark_plot(results)
        print(f"END running {data_name}")
    return
