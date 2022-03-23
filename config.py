"""
All experiments settings should be controlled in this file
"""
import numpy as np

# problem to solve
problem_type = 'classification'  # type of problem, supports 'classification' and 'regression'.
loss = 'Logistic'  # str, loss function name; support "Logistic" for classification; "L2" or "PseudoHuber" for regression
regularizer = "L2"  # str, regularizer type name; support "L2" or "PseudoHuber"
reg_parameter = lambda n: 1./np.sqrt(n) # the regularization parameter multiplying the regularizer. Can be a float, or a lambda function depending on the number of data
dataset_names=["dummy"]
dataset_paths=["datasets/dummy.txt"]

# solvers we want to benchmark
solvers_to_run = ["SVRG","SAN"] # see the classes in src.solvers.whatever.py
solvers_to_load = [""]  # list of algorithm names whose results are ready to load (make sure the results exist).

# parameters controlling how we run the algorithms
n_repetition = 5  # number of repetitions run for stochastic algorithm
epochs = 15  # number of epochs to run for one algorithm, i.e., effective data passes
lr = 1e-2  # float, learning rate for SAN/SANA. default: 1.0
tol = 1e-8  # float, the algorithms will be stopped when the norm of gradient reaches below this threshold.
initialization = "zeros" # default. Could pass a lambda function as well, or "random" (not implemented)
distribution = "uniform"  # for some algorithms, we need to sample data wrt a distribution. Can be "uniform". Other wise must be a lambda function taking nb_data as an input and returning a certain np.array
# example : lambda nb_data, p0=1./(nb_data + 1): np.array([p0] + [(1 - p0) / nb_data] * nb_data)

# parameters controlling the display of the results
measure_time = True # do we measure time elapsed for each epoch? If so, each quantity to be plotted will be plot with respect to both [number of epochs] and [time].
dpi = 50 #1200
plot_threshold = 1e-12 # value under which we stop plotting results (typically machine precision, but also for handling cases when one solver converges way too fast with respect to the others). None if we don't want to threshold at all.
subopt = 1  # Should we plot sub-optimality curves? 1 for yes, 0 for no. Default is 0.

# verbose
save = True # do we save the results?
verbose = 1  # Should we save the outputs (data and plots)? 1 for yes, 0 for no. Default is 1. CONFLICTS HERE, needs cleaning
output_path = 'output'  # folder path to store experiments results
log_file = 'log.txt'  # log file name