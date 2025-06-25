import torchvision

from ab.nn.util.Const import data_dir

__norm_mean = (0.5071, 0.4867, 0.4408)
__norm_dev = (0.2675, 0.2565, 0.2761)

__class_quantity = 100
minimum_accuracy = 1.0 / __class_quantity

def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    # Make sure data_dir points to the folder containing the .tar.gz file (e.g., './data')
    data_dir = './data'
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=False)

    # Also change it for the test set
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=False)
    return (__class_quantity,), minimum_accuracy, train_set, test_set