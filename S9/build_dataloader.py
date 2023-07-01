from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CustomCIFAR10Dataset, Transforms
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout

BatchSize = 128

atrain_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        ShiftScaleRotate(),
        CoarseDropout(max_holes=1, max_height=16,max_width=16,min_holes=1,min_height=16, min_width = 16),
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261],
        ),
        ToTensorV2(),
    ]
)

atest_transform = A.Compose(
    [
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261],
        ),
        ToTensorV2(),
    ]
)
        
def get_train_loader(data_dir,
                     train,
                     download,
                     shuffle,
                     batch_size,
                     transform=None):
    if transform == None:
        transform = Transforms(atrain_transform)
    train_dataset = CustomCIFAR10Dataset(root=data_dir,
                                    transform=transform,
                                    train=train,
                                    download=download)
    
    train_loader = DataLoader(dataset=train_dataset,
                          shuffle=shuffle,
                          batch_size=batch_size)
    
    return train_loader

def get_test_loader(data_dir,
                     train,
                     download,
                     shuffle,
                     batch_size,
                     transform=None):
    if transform == None:
        transform = Transforms(atest_transform)
    test_dataset = CustomCIFAR10Dataset(root=data_dir,
                                    train=train,
                                    transform=transform,
                                    download=download)
    
    test_loader = DataLoader(dataset=test_dataset,
                          shuffle=shuffle,
                          batch_size=batch_size)
    
    return test_loader