import torchvision.transforms as transforms

def transform(_):
    return transforms.Compose([
        transforms.ToTensor()])