from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CustomCIFAR10Dataset, Transforms
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.augmentations.dropout.cutout import Cutout
from albumentations.augmentations.crops.transforms import CropAndPad 

atrain_transform = A.Compose(
    [
        # A.Resize(width=512, height=512),
        # A.RandomCrop(width=300, height=300),
        # A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        # A.HorizontalFlip(p=0.5),
        # ShiftScaleRotate(),
        CropAndPad(px=32,  pad_cval=4, keep_size=True, p=0.1),
        Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=0,  p=0.1),
        # A.VerticalFlip(p=0.1),
        # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        # A.OneOf([
        #     A.Blur(blur_limit=3, p=0.5),
        #     A.ColorJitter(p=0.5),
        # ], p=1.0),
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