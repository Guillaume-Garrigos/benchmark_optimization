# Structure of the classes

This is a brief summary of how the code works and where are its main components

## Problem and Parameters

config.py contains parameters which can be modified by the user
Some datasets should be stored somewhere

- Data : class
    - a fancy dict containing the dataset and its parameters (name, size etc)
    - needed to define the Problem and the Parameters (e.g. we need the dataset name for the title of the plots)
- Parameters : class
    - again a fancy dict. 
    - Contains mostly what's in config.py
    - extracts some information from Data (name, size). The idea is that the heavy dataset itself remains in Data while Parameters is lightweigth (contains only strings, or 1D values)
    - will be used by the solvers (parameters of the algorithm) or when plotting
    - the parameters for the plot are stored in a subclass Parameters.plot (I think that I had a good reason for that)
- Problem : class
    - should contain everything needed to define the problem
    - Contains the loss, regularizer and reg.parameter (see problem.data etc) 
    - provides easy handles to function/gradient so the solvers are easy to read
    - Contains the Data() in Problem.data
        - Right now all our problems depend on Data(), so Problems() is init with Data()
        - This could be changed in the future (for non data-driven problems)
    - Contains all the parameters needed to run the solvers and plot/save the results


## Solver, Records and Results

- Solver : class
    - can be declined in many subclasses, each defining a specific solver
    - Each solver can be different from the others, but here they share:
        - Storing some essential information such as 
            - the Problem() via Solver.problem
            - the Parameters() via Solver.param
        - Running one epoch via .run_epoch 
        - Recording some values of iterest to be plotted later
    - Those values are stored in Solver.records which is a dict { Records.name : Records }
      See the class Records() for more details
- Records : class
    - Records.solver_name points to Solver.name
    - Records.data_name points to Problem.data.name
    - Records.plot points to Plot_param()
    - Records.value is a list which is updated while running an algorithm.
      When one run is done it is emptied and transferred into 
    - Records.value_repetition, which is a list of lists. This will eventually be processed (extracting the mean for the plot etc)
    - the method .save saves one record, and can be laoded later.
- Results : class
    - a fancy 2D dict for which each coefficient is a Records
      corresponding to a specific couple (record_name, solver_name)
    - comes with a couple of methods for plotting, saving the results