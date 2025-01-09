import torchvision.transforms as transforms

def transform(_):
    return transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()])