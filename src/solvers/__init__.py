from pathlib import Path

# the code below imports all the subclasses of Solver() defined in the /solvers folder, possibly accross differnt files
# once this is done, they all exist for the python interpreter
# and they can be accessed in Solver.__subclasses__()
for file in Path("src/solvers/").glob("*.py"):
    module_name = file.stem # basically the name of the file
    if (not module_name.startswith("_")) and (module_name not in ["solver", "records"]):
        __import__('src.solvers.'+module_name, fromlist=['Solver'])
