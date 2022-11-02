import os
import logging
import time

from src.parameters import get_config
from src.problem.data import Data
from src.problem.problem import Problem
from src.results.results import Results
from src.solvers import Solver # this does implicit things in its __init__.py

def get_list_implemented_solvers():
    return [ algo for algo in Solver.__subclasses__() if algo.__name__[0] != '_' ]

def run_solvers(problem):
    """ given a Problem(), run multiple solvers against it, and return Results() """
    # setup the logging and outputing
    logging.basicConfig(
        filename=os.path.join(problem.param.output_folder, problem.param.log_file),
        level=logging.INFO, format='%(message)s')
    logging.info(time.ctime(time.time()))
    # Run solvers
    print(f"Running each solver on {problem.data.name}")
    results = Results(problem) # an empty 2D dict, each coefficient will be a Record
    # Get a list of the available solvers. Nasty but at least we do not need to come back here if create a new solver.
    list_solvers = get_list_implemented_solvers()
    # now we run each solver and stack the results
    for algo in list_solvers:
        # algo.name accessible as a class variable (even before init)
        if algo.name in problem.param.solvers_to_run : # we want to run it
            print(f"Running solver '{algo.name}' on {problem.data.name}")
            solver = algo(problem) # Instanciate a solver
            solver.run_repetition() # run the solver with repetitions
            results.set_records(solver.name, solver.records) # stores all the records from solver.records
    # process the values (extracts mean, min, max), make it ready for analysis
    # and provide each record with an alternate .xaxis_time to plot wrt time
    print("Process the results")
    results.process_values(problem.param.measure_time)
    # eventually saves each records as a file into the folder defined in config
    results.save()
    logging.shutdown() # otherwise python keeps its hands on the log .txt file
    return results


def benchmark_plot(results):
    # We eventually complete results with records to load
    # we assume each record is loaded with a following path:
    #   output_directory/data_name/solver_name-record_name
    results.load(results.param.solvers_to_load)
    results.plot_all(xaxis_time = results.param.measure_time)
    return

def benchmark_datasets():
    config = get_config()
    # loop over all datasets
    for dataset_name in config['problem']['dataset_names']:
        print(f"START running {dataset_name}")
        # setup the main components of the problem, depending on the dataset
        data = Data()
        data.read(dataset_name, config['problem']['dataset_path_folder']) # gathers the components of the data
        problem = Problem(data) # gathers the components defining the problem
        # solve the problem with multiple solvers
        results = run_solvers(problem)
        # plot the results
        if problem.param.do_we_plot:
            benchmark_plot(results)
        print(f"END running {dataset_name}")
    return
