import lightning as L
import torch
import torch.nn as nn
import timm
from torchmetrics.classification import BinaryAccuracy
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from peft import LoraConfig, get_peft_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huggingface_hub import login
from transformers import AutoModel

try:
    login(token="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")
    print("c'est bon")
except:
    print("⚠️ Attention : Token HF non trouvé, l'accès à Curia pourrait échouer.")

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [batch_size, dim] 
        # labels: [batch_size]
        device = features.device
        batch_size = features.shape[0]
        labels = labels.view(-1, 1)
        
        # Création du masque des positifs (même classe)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Calcul de la similarité cosinus (produit scalaire sur vecteurs normalisés)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # Pour la stabilité numérique (Log-Sum-Exp trick)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # On retire l'auto-similarité (la diagonale)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Calcul du Log-Softmax sur les positifs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Moyenne sur tous les positifs pour chaque ancre
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        loss = -mean_log_prob_pos.mean()
        return loss
    
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
    
class CuriaModule(L.LightningModule):
    def __init__(self, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = AutoModel.from_pretrained("raidium/curia")
        config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["query", "value"], # Cibles standards pour ViT
            lora_dropout=0.1,
            bias="none"
        )
        self.model = get_peft_model(self.backbone, config)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.focal_loss = BinaryFocalLossWithSmoothing()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        outputs = self.model(x)
        cls_features = outputs.last_hidden_state[:, 0, :] 
        return self.classifier(cls_features).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1)

        logits = self.forward(x)
        loss = self.focal_loss(logits, y)
        
        self.train_acc(torch.sigmoid(logits), y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1)

        logits = self.forward(x)
        loss = self.focal_loss(logits, y)
        
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

class ContrastiveHDFFModule(L.LightningModule):
    def __init__(self, lr=2e-5, pretraining = True, proj_dim = 128):
        super().__init__()

        self.save_hyperparameters()
        self.pretraining = pretraining

        self.encoder_convnext = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0)
        self.encoder_effnet = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)

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

        self.projection_head = nn.Sequential(
            nn.Linear(total_feats, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim) 
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Premier Dropout
            nn.Linear(total_feats, 1000), # Dense 1000
            nn.ReLU(),
            nn.Dropout(0.5), # Deuxième Dropout (Fig 3: Dropout Layer H2)
            nn.Linear(1000, 1) # Output Prediction (N classes = 1 pour binaire)
        )

        self.scl_loss = SupConLoss(temperature=0.1)
        self.focal_loss = BinaryFocalLossWithSmoothing(alpha=0.25, gamma=2, smoothing=0.1)

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
    def get_features(self, x):
        f_conv = self.encoder_convnext(x)
        f_eff = self.encoder_effnet(x)
        combined = torch.cat((f_conv, f_eff), dim=1)
        return combined 
    
    def forward(self, x):
        features = self.get_features(x)
        if self.pretraining:
            z = self.projection_head(features)
            return F.normalize(z, dim=1)  # Normalisation pour SupConLoss
        else: ## classification
            return self.classifier(features).squeeze(-1)
   

    def training_step(self, batch):
        x, y, _ = batch
        y = y.squeeze(-1)

        if self.pretraining:
            z = self.forward(x)
            loss = self.scl_loss(z, y)
            self.log("train/scl_loss", loss, prog_bar=True)
        else:
            logits = self.forward(x)
            loss = self.focal_loss(logits, y)
            self.train_acc(torch.sigmoid(logits), y)
            self.log("train/focal_loss", loss, prog_bar=True)
            self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        

        return loss

    def validation_step(self, batch):
        x, y, _ = batch
        y = y.squeeze(-1)

        if self.pretraining:
            z = self.forward(x)
            loss = self.scl_loss(z, y)
            self.log("val/scl_loss", loss, prog_bar=True)
        else:
            logits = self.forward(x)
            loss =self.focal_loss(logits, y)
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
                pct_start=0.1,    # 10% du temps en Warmup
                anneal_strategy='cos',
                div_factor=3000.0,  # LR de départ = max_lr / 25
                final_div_factor=1.0 # LR final très bas pour la précision
            )        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # On surveille l'accuracy pour le scheduler
                "frequency": 1,
            }
        }
          
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

    def cutmix_data(self, x, y, alpha=1.0):
        """Applique le CutMix sur un batch."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        # Tirage aléatoire de la boîte de découpe
        W = x.size()[2]
        H = x.size()[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Centre de la boîte
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # On remplace la zone par celle de l'image indexée
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # On ajuste lambda pour correspondre à la surface réelle découpée
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
    
    def training_step(self, batch):
        x, y, _ = batch    
        y = y.view(-1).float() # [B, 1] -> [B] et conversion en float pour BCEWithLogits
        # Log metrics
        if self.current_epoch < self.trainer.max_epochs - 5 and np.random.rand() > 0.2:
            x, y_a, y_b, lam = self.cutmix_data(x, y, alpha=1.0)
            logits = self(x).view(-1)
            # La perte est la combinaison linéaire des pertes des deux labels
            loss = lam * self.criterion(logits, y_a.view(-1)) + (1 - lam) * self.criterion(logits, y_b.view(-1))
        else:
            # Entraînement standard (ou fin d'entraînement sans CutMix pour stabiliser)
            logits = self(x).view(-1)
            loss = self.criterion(logits, y.view(-1))

        self.train_acc(torch.sigmoid(logits), y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
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
    
