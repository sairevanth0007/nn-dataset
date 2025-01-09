import torchvision.transforms as transforms


def transform(norm):
    return transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])