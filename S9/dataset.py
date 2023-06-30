from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import numpy as np
import albumentations as A

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root, transform, train, download):
        super().__init__()
        self.cifar10_dataset = CIFAR10(root=root,
                                       train=train,
                                       download=download)
        self.transform = transform
    def __len__(self):
        return len(self.cifar10_dataset)
    
    def __getitem__(self, index):
        image, label = self.cifar10_dataset[index]
        if self.transform:
            image = self.transform(image)
        return image,label
    
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]