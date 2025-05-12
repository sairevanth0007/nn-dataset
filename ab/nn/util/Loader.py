from ab.nn.util.Util import get_ab_nn_attr

def get_obj(name, o_type):
    """ Dynamically load a function/field by name if provided from the object of type 'o_type'"""
    return get_ab_nn_attr(f"{o_type}.{name}", o_type)


def load_dataset(task, dataset_name, transform_name):
    """
    Dynamically load dataset and transformation based on the provided paths.
    :param task: Task name
    :param dataset_name: Dataset name
    :param transform_name: Transform name
    :return: Train and test datasets.
    """
    loader = get_obj(dataset_name, 'loader')
    # Call the loader function with the dynamically loaded transform
    return loader(get_obj(transform_name, 'transform'), task)
