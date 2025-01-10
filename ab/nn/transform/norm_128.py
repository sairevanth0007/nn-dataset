import torchvision.transforms as transforms


def transform(norm):
    return transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])