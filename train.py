import os
import argparse
import h5py
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging
from torch.utils.data import DataLoader

from data import HistopathDataset, get_transforms
from models import HDFFModule

def train(args):

    with h5py.File(args.train_path, 'r') as f:
        ref_img = f[list(f.keys())[10]]['img'][()] # L'index 10 au hasard
        if ref_img.shape[0] == 3:
            ref_img = ref_img.transpose(1, 2, 0)
    train_ds = HistopathDataset(args.train_path, transforms=get_transforms(mode='train'), ref_image=ref_img)
    val_ds = HistopathDataset(args.val_path, transforms=get_transforms(mode='val'), ref_image=ref_img)

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # Model Initialization
    model = HDFFModule(
        lr=args.lr
    )

    # Callbacks and Logger
    checkpoint_dir = os.path.join("models", "hdff_model")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-model-{epoch:02d}-{val_acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/acc",
        patience=7,
        mode="max"
    )

    logger = TensorBoardLogger("logs", name="hdff_model")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto", # Use GPU
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar(), lr_monitor, StochasticWeightAveraging(swa_lrs=1e-6)],
        logger=logger,
        precision="16-mixed" if args.use_fp16 else 32,
        deterministic=True
    )
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"♻️ Resuming training from checkpoint: {args.resume_from}")
        ckpt_path = args.resume_from
    else:
        if args.resume_from:
            print(f"⚠️ Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")
        ckpt_path = None

    print(f"Training started")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path (do not change the default paths, they are set to work with the provided data structure)
    parser.add_argument("--train_path", type=str, default="data/train.h5")
    parser.add_argument("--val_path", type=str, default="data/val.h5")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    
    # Hardware
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_fp16", action="store_true", help="Use 16-bit precision")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint (.ckpt) to resume from")

    args = parser.parse_args()
    train(args)