# Documentation

Attempt to make this code usable.

## Running a grid search

We can set the config file to run grid search over some parameter. 
Suppose that we want to run the SGD solver, with some specific parameters: the stepsize will be decreasing and have the form $\frac{C}{t^\alpha}$, with $C=2$ and $\alpha = \frac{1}{2}$. This can be done as follows:

```
solvers: 
    - SGD:
        stepsize_type: vanishing
        stepsize_factor: 2.0
        stepsize_vanishing_exponent: 0.5
```

Suppose now that we want to change the value of $C$ and do a **linesearch** over this parameter. This can be done thanks to the `grid_search` key:

```
solvers: 
    - SGD:
        stepsize_type: vanishing
        stepsize_vanishing_exponent: 0.5
        grid_search:
            stepsize_factor:
                - 0.001
                - 0.01
                - 0.1
                - 1.0
```

If you want to do a grid search over a large number of parameters, entering them one by one could be tedious, so it is also possible to simply set some parameters defining the grid. You must set the minimal and maximal values, and how many parameter you want. By default the grid is taken on a linear scale, but you can also choose logarithmic one.

```
solvers: 
    - SGD:
        stepsize_type: vanishing
        grid_search:
            stepsize_factor:
                min: 0.001
                max: 1.0
                number: 40
                scale: log # linear
```

Of course it is possible to compare two different solvers using the same parameters. For instance with:

```
solvers: 
    - SGD:
        stepsize_type: constant
        grid_search:
            stepsize_factor:
                min: 0.0001
                max: 10.0
                number: 30
                scale: log
    - GD:
        stepsize_type: constant
        grid_search:
            stepsize_factor:
                min: 0.0001
                max: 10.0
                number: 30
                scale: log
```

Note that, by default, solvers run with a grid search won't appear when plotting curves for Records (gradient norm, etc). This is because there might be way too much curves and it would be a mess. If you want to override this, set

```
results: 
    grid_search_curves_plot: True
```

A few other options are available :
- label : a string, which modifies what is displayed as the name of the parameter in the figure. By default it is the hard-coded name of the variable.
- title : a string, which is going to be used as the title of the figure. Can be set to None to have no title.