import torchvision.transforms as transforms

def transform(_):
    return transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])