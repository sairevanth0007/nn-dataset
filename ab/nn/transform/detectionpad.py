import torchvision.transforms as transforms
import torch.nn.functional as F

class PadToSize:
    def __init__(self, max_size=(800, 800)):
        """
        Pad images to a fixed size (height, width).

        Parameters:
        -----------
        max_size : tuple, optional
            Maximum size (height, width) to pad images to. Default is (800, 800).
        """
        self.max_size = max_size

    def __call__(self, image):
        """
        Apply padding to the image.

        Parameters:
        -----------
        image : torch.Tensor
            Input image tensor of shape (C, H, W).

        Returns:
        --------
        torch.Tensor
            Padded image tensor.
        """
        _, h, w = image.shape
        max_h, max_w = self.max_size
        pad_h = max_h - h
        pad_w = max_w - w

        # Apply padding if necessary
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        return image

def transform(_):
    max_size=(800, 800)
    """
    Returns a transform pipeline for object detection.

    Parameters:
    -----------
    max_size : tuple, optional
        Maximum size (height, width) to pad images to. Default is (800, 800).
    **kwargs : dict
        Additional arguments (unused in this implementation).

    Returns:
    --------
    transforms.Compose
        A composition of transforms including padding.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
        PadToSize(max_size)     # Pad to the specified size
    ])
