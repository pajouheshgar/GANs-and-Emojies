import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    n_jobs = int(os.environ['NUMBER_OF_PROCESSORS']) - 1
except:
    pass
data_dir = "../../Dataset/"


def print_function(inp, file=None):
    if file is not None:
        if os.path.exists(file):
            mode = 'a'
        else:
            mode = 'w'
        with open(file, mode=mode) as f:
            f.write(inp + "\n")

    print(inp)
