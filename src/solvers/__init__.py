from pathlib import Path

def is_forbidden_filename(string):
    return (string.startswith("_") # ignore __init__ or others
            or (string in ["solver", "records"]) # those contain the definition of the classes
            or string.endswith("-checkpoint") ) # those can appear with jupyterlab in .ipynb_checkpoints

def is_forbidden_foldername(string):
    return (string.startswith("_") # ignore __pycache__ or others
            or string.startswith(".")) # ignore .ipynb_checkpoints or others

def import_all_classes(folder_path, class_name):
    """ Import all the classes of type class_name
        that can be found within folder_path
        There is no recursion here, we only look at files like folder_path/file_name.py
    """
    for file in Path(folder_path).glob(r"**/*.py"): # any .py file in any subfolder
        file_name = file.stem # the name of the file without folder or extension
        if not is_forbidden_filename(file_name): 
            # file > remove extension > replace / with .
            file_import_path = ".".join(part for part in file.with_suffix('').parts) 
            __import__(file_import_path, fromlist=[class_name])
            
# the code below imports all the subclasses of Solver() defined in the /solvers folder, possibly accross differnt files
# once this is done, they all exist for the python interpreter
# and they can be accessed in Solver.__subclasses__()
import_all_classes(folder_path="src/solvers/", class_name='Solver')
