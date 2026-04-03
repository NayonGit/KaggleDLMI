import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HistopathDataset(Dataset):
    def __init__(self, dataset_path, transforms=None, mode='train'):
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.mode = mode
        self.hdf = h5py.File(dataset_path, 'r')
        self.image_ids = list(self.hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img_id = self.image_ids[idx]
        group = self.hdf[img_id]
        
        # Image (HWC, uint8)
        img = np.array(group['img'], dtype=np.float32).transpose(1, 2, 0)
        if img.max() <= 1.01:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
   
        img = np.ascontiguousarray(img)

        # Label 
        if 'label' in group:
            raw_label = group['label']
            label_val = np.array(raw_label).item()
        else:
            # Si le label n'existe pas (Test Set Kaggle)
            label_val = -1.0
        
        
        # Metadata 
        if 'metadata' in group:
            raw_meta = group['metadata']
            center_val = int(np.array(raw_meta)[0])
        else:
            center_val = -1 # Valeur par défaut si pas de metadata

        # Transformations
        if self.transforms:
            augmented = self.transforms(image=img)
            img_tensor = augmented['image']
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # label_val devient un tenseur de shape (1,)
        return img_tensor, torch.tensor([label_val], dtype=torch.float32), int(img_id)
    
def get_transforms(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5),            
            A.Transpose(p=0.5),

            A.HueSaturationValue(
                hue_shift_limit=15, 
                sat_shift_limit=35, 
                val_shift_limit=15, 
                p=0.6
            ),

            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.1), 
                contrast_limit=0.2, 
                p=0.5
            ),

            # A.OneOf([
            #     A.GaussianBlur(blur_limit=(3, 3), p=0.5),
            #     A.GaussNoise(std_range=(0.02,0.05), p=0.3),
            # ], p=0.3),
            

            # A.RandomResizedCrop(
            #     size=(96, 96), 
            #     scale=(0.8, 1.0), 
            #     ratio =(0.9, 1.1),
            #     p=0.5
            # ),          
                          
            # A.CoarseDropout(
            #     num_holes_range=(4, 6), 
            #     hole_height_range=(10, 12), 
            #     hole_width_range=(10, 12), 
            #     p=0.3
            # ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

def get_transforms_curia(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.Resize(512, 512, interpolation = cv2.INTER_CUBIC), 
            A.ToGray(num_output_channels = 1, p=1.0),
            
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-45, 45), shear=(-20, 20), p=0.2),
            
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(512,512, interpolation = cv2.INTER_CUBIC), 
            A.ToGray(num_output_channels = 1, p=1.0),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])

def get_transforms_dinov2(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.Resize(98, 98, interpolation = cv2.INTER_CUBIC), 
            
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),

            A.HueSaturationValue(
                hue_shift_limit=15, 
                sat_shift_limit=35, 
                val_shift_limit=15, 
                p=0.6
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.1), 
                contrast_limit=0.2, 
                p=0.5
            ),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 3), p=0.5),
                A.GaussNoise(std_range=(0.02,0.05), p=0.3),
            ], p=0.3),
            

            A.RandomResizedCrop(
                size=(96, 96), 
                scale=(0.8, 1.0), 
                ratio =(0.9, 1.1),
                p=0.5
            ),          
                          
            A.CoarseDropout(
                num_holes_range=(4, 6), 
                hole_height_range=(10, 12), 
                hole_width_range=(10, 12), 
                p=0.3
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
                ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(98,98, interpolation = cv2.INTER_CUBIC), 
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
                ),         
            ToTensorV2(),
        ])