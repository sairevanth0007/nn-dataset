from ab.nn.util.Util import get_attr

class Loader:
    @staticmethod
    def load_dataset(task, dataset_name, transform_name):
        """
        Dynamically load dataset and transformation based on the provided paths.
        :param task: Task name
        :param dataset_name: Dataset name
        :param transform_name: Transform name
        :return: Train and test datasets.
        """
        # Dynamically load the transform function if provided
        t_nm = 'transform'
        transform_fn = get_attr(f"{t_nm}.{transform_name}", t_nm)

        # Dynamically load the loader function
        l_nm = 'loader'
        loader = get_attr(f"{l_nm}.{dataset_name}", l_nm)
        # Call the loader function with the dynamically loaded transform
        return loader(transform_fn, task)
