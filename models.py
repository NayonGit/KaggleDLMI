import lightning as L
import torch
import torch.nn as nn
import timm
from torchmetrics.classification import BinaryAccuracy
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from peft import LoraConfig, get_peft_model

class BinaryFocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=1, gamma=2, smoothing=0.1):
        super(BinaryFocalLossWithSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Label Smoothing to avoid overconfidence and overfitting
        # [0, 1] -> [0.05, 0.95]
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Standard BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction='none')
        
        # Focal Loss = alpha * (1 - pt)^gamma * BCE
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        return focal_loss.mean()
    
class HDFFModule(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()

        self.save_hyperparameters()
        
        self.encoder_convnext = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0)
        self.encoder_effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

        # for param in self.encoder_convnext.parameters():
        #     param.requires_grad = False
        # for param in self.encoder_effnet.parameters():
        #     param.requires_grad = False

        dummy_x = torch.randn(1, 3, 96, 96)
        with torch.no_grad():
            feat_convnext = self.encoder_convnext(dummy_x).shape[1] # 768 pour Tiny
            feat_effnet = self.encoder_effnet(dummy_x).shape[1]   # 1408 pour B2
        total_feats = feat_convnext + feat_effnet
        print(f"[Info] Dual-Model Fusion: {total_feats} features ({feat_convnext} + {feat_effnet})")

        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Premier Dropout
            nn.Linear(total_feats, 1000), # Dense 1000
            nn.ReLU(),
            nn.Dropout(0.5), # Deuxième Dropout (Fig 3: Dropout Layer H2)
            nn.Linear(1000, 1) # Output Prediction (N classes = 1 pour binaire)
        )
        self.loss_fn = BinaryFocalLossWithSmoothing(alpha=0.25, gamma=2, smoothing=0.1)
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        f_conv = self.encoder_convnext(x)
        f_eff = self.encoder_effnet(x)

        # we combine them: hdff
        combined = torch.cat((f_conv, f_eff), dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(-1)
   
    
    def training_step(self, batch):
        x, y, _ = batch
        y = y.squeeze(-1)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Log metrics
        self.train_acc(torch.sigmoid(logits), y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch):
        x, y, _ = batch
        y = y.squeeze(-1)
        logits = self(x)
        loss = self.loss_fn(logits, y)    

        self.val_acc(torch.sigmoid(logits), y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.3)
        scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.2,    # 10% du temps en Warmup
                anneal_strategy='cos',
                div_factor=25.0,  # LR de départ = max_lr / 25
                final_div_factor=1000.0 # LR final très bas pour la précision
            )        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # On surveille l'accuracy pour le scheduler
                "frequency": 1,
            }
        }
def get_model(model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k", method: str = "dora", r: int = 16):
    """
    ConvNeXtV2 with PEFT method
    """
    base_model = timm.create_model(model_name, pretrained=True, num_classes=1)

    if method == "full":
        print(f"[Info] Full Fine-Tuning : {model_name}")
        return base_model

    if "convnext" in model_name:
        target_modules = ["fc1", "fc2"]
    else:
        # Pour ResNet ou EfficientNet (ajustable selon le backbone choisi)
        target_modules = ["conv_pw", "conv_pwl"]

    print(f"[PEFT] Application de {method.upper()} (r={r})")
    
    config = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        use_dora=(method == "dora"),
        # 'head' est le nom standard de la couche finale dans timm pour ConvNeXt
        modules_to_save=["head"], 
    )

    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()
    
    return model

class HistopathLightningModule(L.LightningModule):
    def __init__(self, model_name="convnextv2_tiny.fcmae_ft_in22k_in1k", method="dora", r=16, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = get_model(model_name, method, r)

        self.criterion = BinaryFocalLossWithSmoothing(alpha=1, gamma=2, smoothing=0.1)
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        self.train_acc(torch.sigmoid(logits), y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)    

        self.val_acc(torch.sigmoid(logits), y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.05)
        # On utilise un scheduler qui réduit le LR quand la loss de val stagne
        scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,    # 10% du temps en Warmup
                anneal_strategy='cos',
                div_factor=25.0,  # LR de départ = max_lr / 25
                final_div_factor=1000.0 # LR final très bas pour la précision
            )        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # On surveille l'accuracy pour le scheduler
                "frequency": 1,
            }
        }