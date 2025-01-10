import torchvision.transforms as transforms

def transform(_):
    return transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()])