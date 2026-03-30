import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ReinhardNormalizer:
    def __init__(self):
        self.target_mean = None
        self.target_std = None

    def fit(self, target_image):
        """Calcule les stats de référence sur une image type."""
        # OpenCV utilise BGR par défaut, on convertit RGB -> LAB
        lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.target_mean = np.mean(lab, axis=(0, 1))
        self.target_std = np.std(lab, axis=(0, 1))

    def transform(self, image):
        """Applique le transfert de stats à une nouvelle image."""
        if self.target_mean is None: 
            return image
            
        # Conversion image source RGB -> LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        img_mean = np.mean(lab, axis=(0, 1))
        img_std = np.std(lab, axis=(0, 1))
        
        # Transfert de statistiques (Reinhard et al.)
        # Formule : (I - mean_i) * (std_target / std_i) + mean_target
        norm_lab = (lab - img_mean) * (self.target_std / (img_std + 1e-6)) + self.target_mean
        
        # Clip pour rester dans les bornes [0, 255] et retour en RGB
        norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)

class HistopathDataset(Dataset):
    def __init__(self, dataset_path, transforms=None, mode='train',ref_image=None):
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.mode = mode
        self.hdf = h5py.File(dataset_path, 'r')
        self.image_ids = list(self.hdf.keys())
        self.normalizer = None

        if ref_image is not None:
            self.normalizer = ReinhardNormalizer()
            self.normalizer.fit(ref_image)
        

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

        if self.normalizer is not None:
            img = self.normalizer.transform(img)
                
        img = np.ascontiguousarray(img)

        # Label 
        raw_label = group['label']
        label_val = np.array(raw_label).item() # .item() convertit un array de taille 1 en scalaire
        
        # Metadata 
        raw_meta = group['metadata']
        center_val = int(np.array(raw_meta)[0])

        # Transformations
        if self.transforms:
            augmented = self.transforms(image=img)
            img_tensor = augmented['image']
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # label_val devient un tenseur de shape (1,)
        return img_tensor, torch.tensor([label_val], dtype=torch.float32), center_val
    
def get_transforms(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5),            
            A.Transpose(p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.5),
            ], p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])