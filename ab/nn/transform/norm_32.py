import torchvision.transforms as transforms


def transform(norm):
    return transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])