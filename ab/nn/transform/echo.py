import torchvision.transforms as transforms

def transform():
    return transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()])