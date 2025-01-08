from os.path import join
from pathlib import Path

to_nn = ('ab', 'nn')

default_config = ''
default_epochs = 1
default_trials = -1 # one more trial
default_min_batch_power = 0
default_max_batch_power = 12
default_min_lr = 1e-5
default_max_lr = 1.0
default_min_momentum = 0.0
default_max_momentum = 1.0
default_nn_fail_attempts = 30
default_random_config_order = False
default_transform = None

minimum_accuracy_multiplayer =  1.2


def nn_path(dr):
    """
    Defines path to ab/nn directory.
    """
    import ab.nn.__init__ as init_file
    return Path(init_file.__file__).parent.absolute() / dr


metric_dir = nn_path('metric')
nn_dir = nn_path('nn')
transform_dir = nn_path('transform')

stat_dir = nn_path('stat')


def __to_root_paths():
    """
    Defines path to the project root directory.
    """
    project_root = '.'
    if nn_path('') == Path().absolute():
        project_root = ['..'] * len(to_nn)
    return project_root

__project_root_path_list = __to_root_paths()
data_dir = join(*__project_root_path_list, 'data')
db_dir = join(*__project_root_path_list, 'db')
db_file = join(db_dir, 'ab.nn.db')

main_tables = ('stat',)
code_tables = ('nn', 'transform', 'metric')
param_tables = ('prm',)
dependent_columns = code_tables + param_tables
all_tables = code_tables + main_tables + param_tables
index_colum = ('task', 'dataset') + dependent_columns
extra_main_columns = ('duration', 'accuracy')