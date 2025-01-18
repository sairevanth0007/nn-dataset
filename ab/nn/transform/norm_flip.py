import torchvision.transforms as transforms


def transform(norm):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])
