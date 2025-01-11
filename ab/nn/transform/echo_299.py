import torchvision.transforms as transforms

def transform(_):
    return transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor()])