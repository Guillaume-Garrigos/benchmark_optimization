import os
import logging
import time

from src.parameters import get_config
from src.problem.data import Data
from src.problem.problem import Problem
from src.results.results import Results
from src.solvers import Solver # this does implicit things in its __init__.py

def get_dict_implemented_solvers():
    return { Algo.name : Algo  for Algo in Solver.__subclasses__() if Algo.__name__[0] != '_' }

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
    dict_solvers = get_dict_implemented_solvers()
    # now we run each solver and stack the results
    for flavor in problem.param.solvers:
        # flavor = { algo_name : dict_of_parameters }
        algo_name = list(flavor.keys())[0]
        algo_param = flavor[algo_name]
        flavor_name = algo_param['flavor_name'] # could be different from algo_name if running multiple flavors
        # sanity check that the name given by the user is implemented
        if algo_name not in dict_solvers.keys():
            print(f"Warning: The Solver '{algo_name}' is not implemented. This will be skipped.")
        else:
            Algo = dict_solvers[algo_name] # we get the handle on the desired solver
            print(f"Running solver '{algo_name}' with flavor '{flavor_name}' on the dataset '{problem.data.name}'")
            solver = Algo(problem, algo_param) # Instanciate a solver with flavoured parameters
            solver.run_repetition() # run the solver with repetitions
            results.set_records(flavor_name, solver.records) # stores all the records from solver.records
    # run is over
    # process the values (extracts mean, min, max), make it ready for analysis
    # provide each record with an alternate .xaxis_time to plot wrt time (optional)
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
