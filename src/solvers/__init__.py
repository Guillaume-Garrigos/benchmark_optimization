from src.utils import import_all_classes
            
# the code below imports all the subclasses of Solver() defined in the /solvers folder, possibly accross differnt files
# once this is done, they all exist for the python interpreter
# and they can be accessed in Solver.__subclasses__()
import_all_classes(folder_path="src/solvers/", class_name='Solver')
