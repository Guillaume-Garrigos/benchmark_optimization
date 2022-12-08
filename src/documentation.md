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

It is also possible to perform a lineasearch on more than one parameter, but beware that the cost increases exponentially.
In our example, we could explore different values for $\alpha$:


```
solvers: 
    - SGD:
        stepsize_type: vanishing
        grid_search:
            stepsize_factor:
                - 0.001
                - 0.01
                - 0.1
                - 1.0
            stepsize_vanishing_exponent:
                - 0.25
                - 0.5
                - 0.75
```