import lightning as L

from peft import LoraConfig, get_peft_model

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy
from torch.optim.lr_scheduler import OneCycleLR

from huggingface_hub import login
from transformers import AutoModel, AutoImageProcessor


try:
    login(token="YOUR_TOKEN")
    print("Registration Successful: You can now access Hugging Face models that require authentication.")
except:
    print("⚠️ Warning : Token HF not found.")


class Dinov2Module(L.LightningModule):
    def __init__(self, lr=5e-5, smoothing=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none"
        )
        self.model = get_peft_model(self.backbone, config)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(768), 
            nn.Linear(768, 1)
        ) ## standard logistic regression

        self.criterion = nn.BCEWithLogitsLoss()
        self.smoothing = smoothing

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
    
    def forward(self, x):
        outputs = self.model(x)
        patch_tokens = outputs.last_hidden_state[:, 1:, :] # cls ignored
        pooled_features = torch.mean(patch_tokens, dim=1)  # mean pooling
        
        return self.classifier(pooled_features).squeeze(-1)

    def training_step(self, batch):
        x, y, _ = batch
        y = y.squeeze(-1).float()
        
        # Label Smoothing
        with torch.no_grad():
            y_smooth = y * (1.0 - self.smoothing) + 0.5 * self.smoothing
            
        logits = self.forward(x)
        loss = self.criterion(logits, y_smooth)
        
        # Log metrics 
        acc = self.train_acc(torch.sigmoid(logits), y.int())
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1)

        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        self.val_acc(torch.sigmoid(logits), y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10.0, 
            final_div_factor=100.0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
class UNI2Module(L.LightningModule):
    def __init__(self, lr=1e-4): # LR plus élevé car on n'entraîne que le classifier
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        self.backbone = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval() 
        

        self.classifier = nn.Sequential(
            nn.Linear(1536, 768),         # Layer 0
            nn.LayerNorm(768),            # Layer 1 
            nn.ReLU(),                    # Layer 2
            nn.Dropout(0.2),              # Layer 3
            nn.Linear(768, 1)             # Layer 4
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x) # [Batch, 1536]
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1).float()
        
        # On garde ton Label Smoothing (0.1) qui aide à la calibration
        smoothing = 0.1
        y_smooth = y * (1.0 - smoothing) + 0.5 * smoothing
        
        logits = self.forward(x).squeeze(-1)
        loss = self.criterion(logits, y_smooth)

        self.train_acc(torch.sigmoid(logits), y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1).float() 
        logits = self.forward(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y) # On garde BCE pour la val  

        self.val_acc(torch.sigmoid(logits), y.int())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.lr,
            weight_decay=0.05
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

class GigaPathModule(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # GigaPath utilise une architecture ViT-L/14 (embed_dim=1536)
        self.backbone = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval() 

        self.classifier = nn.Sequential(
            nn.Linear(1536, 768),         # Layer 0
            nn.LayerNorm(768),            # Layer 1 
            nn.ReLU(),                    # Layer 2
            nn.Dropout(0.2),              # Layer 3
            nn.Linear(768, 1)             # Layer 4
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x) # Shape: [Batch, 1536]
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1).float()
        
        smoothing = 0.1
        y_smooth = y * (1.0 - smoothing) + 0.5 * smoothing
        
        logits = self.forward(x).squeeze(-1)
        loss = self.criterion(logits, y_smooth)

        self.train_acc(torch.sigmoid(logits), y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1).float()
        logits = self.forward(x).squeeze(-1)
        loss = self.criterion(logits, y)
        
        self.val_acc(torch.sigmoid(logits), y.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.lr,
            weight_decay=0.05
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
class PhikonV2Module(L.LightningModule):
    def __init__(self, lr=5e-5, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()
        
        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        self.model = AutoModel.from_pretrained("owkin/phikon-v2")
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1) # Sortie binaire pour ton projet
        )

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        
        cls_features = outputs.last_hidden_state[:, 0, :]
        
        return self.classifier(cls_features)

    def training_step(self, batch):
        x, y, _ = batch
        y = y.squeeze(-1).float() 
        logits = self.forward(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        self.train_acc(torch.sigmoid(logits), y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch):
        x, y, _ = batch
        y = y.squeeze(-1).float() 
        logits = self.forward(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        acc = self.val_acc(torch.sigmoid(logits), y.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]   
    
class Virchow2Module(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2", 
            pretrained=True, 
            mlp_layer=SwiGLUPacked, 
            act_layer=nn.SiLU
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval() 

        ## Linear Probing as the training is a bit different here (we use all tokens)
        self.classifier = nn.Sequential(
            nn.Linear(2560, 1) 
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        with torch.no_grad():
            # Output size: [Batch, 261, 1280]
            # 261 = 1 (class token) + 4 (registers) + 256 (patches)
            output = self.backbone(x) 
            
            class_token = output[:, 0]        # [Batch, 1280]
            patch_tokens = output[:, 5:]      # [Batch, 256, 1280] 
            
            # Global Average Pooling 
            patch_pool = patch_tokens.mean(dim=1) # [Batch, 1280]
            
            embedding = torch.cat([class_token, patch_pool], dim=-1) # [Batch, 2560]
            
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1).float()
        
        smoothing = 0.1
        y_smooth = y * (1.0 - smoothing) + 0.5 * smoothing
        
        logits = self.forward(x).squeeze(-1)
        loss = self.criterion(logits, y_smooth)

        self.train_acc(torch.sigmoid(logits), y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1).float()
        logits = self.forward(x).squeeze(-1)
        loss = self.criterion(logits, y)
        
        self.val_acc(torch.sigmoid(logits), y.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.lr,
            weight_decay=0.1
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
