from ab.nn.util.Util import get_attr

class Loader:
    @staticmethod
    def load_dataset(dataset_name, transform_name, **kwargs):
        """
        Dynamically load dataset and transformation based on the provided paths.
        :param dataset_name: Dataset name
        :param transform_name: Transform name
        :param kwargs: Additional parameters for the loader and transform.
        :return: Train and test datasets.
        """
        # Dynamically load the transform function if provided

        transform_module, transform_func = f"transform.{transform_name}.transform".rsplit('.', 1)
        transform = get_attr(transform_module, transform_func)()

        # Dynamically load the loader function
        loader_module, loader_func = f"loader.{dataset_name}.loader".rsplit('.', 1)
        loader = get_attr(loader_module, loader_func)
        # Call the loader function with the dynamically loaded transform
        return loader(transform=transform, **kwargs)
