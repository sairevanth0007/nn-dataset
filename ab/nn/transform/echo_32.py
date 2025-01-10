import torchvision.transforms as transforms

def transform(_):
    return transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()])