# benchmark_optimization

This is for running benchmarks between optimization solvers. Enmphasis is put on making it easy to add new solvers.

## To get started

1. Clone this repo
    - `git clone --recurse-submodules https://github.com/Guillaume-Garrigos/benchmark_optimization.git`
    - `cd benchmark_optimization`
    - `git submodule foreach --recursive git checkout main`
    - Note that we use the `--recurse-submodules` option here. This is for you to download all the solvers available to you.
2. [optional] Add some specific solvers in `/src/solvers/`
    - you can type your own solvers in a python file, eventually in some subfolder.
3. [optional] Download some datasets (the repo comes with a `dummy` dataset to run first experiments).
    - datasets must be placed in `/datasets/`, with the .txt extension.
    - datasets can be downloaded here : [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).
4. Edit `config.py` to specify which solvers to run, with which parameters. If you want more details about what are the parameters, you can give a look at `/src/config_default.py`
5. Run experiments. You have different options:
    - Run `python run.py` or `bash run.sh`
    - Open a python script or notebook, and run the following lines:
    ```
    from src.benchmark import benchmark_datasets

    benchmark_datasets()
    ```



