import torchvision.transforms as transforms

def transform(_):
    return transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])