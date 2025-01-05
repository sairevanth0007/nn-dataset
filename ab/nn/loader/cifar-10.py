import torchvision

from ab.nn.util.Const import data_dir


def loader(transform=None):
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    return (10,), train_set, test_set