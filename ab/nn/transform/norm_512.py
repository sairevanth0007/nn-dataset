import torchvision.transforms as transforms


def transform(norm):
    return transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])