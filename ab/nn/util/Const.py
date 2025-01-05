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


def __nn_path(dr):
    """
    Defines path to ab/nn directory.
    """
    import ab.nn.__init__ as init_file
    return Path(init_file.__file__).parent.absolute() / dr


metric_dir = __nn_path('metric')
dataset_dir = __nn_path('dataset')
stat_dir = __nn_path('stat')
transform_dir = __nn_path('transform')


def __to_root_paths():
    """
    Defines path to the project root directory.
    """
    project_root = '.'
    if __nn_path('') == Path().absolute():
        project_root = ['..'] * len(to_nn)
    return project_root

__project_root_path_list = __to_root_paths()
data_dir = join(*__project_root_path_list, 'data')
db_dir = join(*__project_root_path_list, join('db', 'ab.nn.db'))
