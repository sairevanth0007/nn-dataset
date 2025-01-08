import torchvision.transforms as transforms


def transform(norm):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*norm)])
