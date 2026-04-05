import os
import argparse
from pyexpat import model
import h5py
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch

from data import HistopathDataset, get_transforms_fms
from models import UNI2Module, PhikonV2Module, Dinov2Module, GigaPathModule, Virchow2Module

MODEL_REGISTRY = {
    "uni2": UNI2Module,
    "phikonv2": PhikonV2Module,
    "dinov2": Dinov2Module,
    "gigapath": GigaPathModule,
    "virchow2": Virchow2Module
}

def train(args):

    torch.set_float32_matmul_precision('high')
    
    train_ds = HistopathDataset(args.train_path, transforms=get_transforms_fms(mode='train'))
    val_ds = HistopathDataset(args.val_path, transforms=get_transforms_fms(mode='val'))

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

    if args.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {args.model_name} unknown. Choose among: {list(MODEL_REGISTRY.keys())}")
    
    print(f"Model Initialization : {args.model_name}")
    model_class = MODEL_REGISTRY[args.model_name]
    model = model_class(lr=args.lr)
    
    # Callbacks and Logger
    checkpoint_dir = os.path.join("models", args.model_name)
    logger_name = f"{args.model_name}_experiment"
    monitor_metric = "val/acc"
    mode = "max"
    filename_format = f"{args.model_name}-{{epoch:02d}}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename_format,
        monitor=monitor_metric,
        mode=mode,
        save_top_k=3,
        save_last=True,
        every_n_epochs = 1
    )
    
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=10,
        mode=mode
    )

    logger = TensorBoardLogger("logs", name=logger_name)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [RichProgressBar(), lr_monitor, checkpoint_callback, early_stop_callback]
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto", # Use GPU
        devices=1,
        callbacks=callbacks,
        logger=logger,
        precision="16-mixed" if args.use_fp16 else 32,
        deterministic=True
    )
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        ckpt_path = args.resume_from
    else:
        if args.resume_from:
            print(f"⚠️ Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")
        ckpt_path = None

    print(f"Training started")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.h5")
    parser.add_argument("--val_path", type=str, default="data/val.h5")
    
    # Model Selection
    parser.add_argument("--model_name", type=str, default="uni2", choices=list(MODEL_REGISTRY.keys()), 
                        help="Model to train. Choose among: " + ", ".join(MODEL_REGISTRY.keys()))
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_fp16", action="store_true", help="Use 16-bit precision")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint (.ckpt) to resume from")

    args = parser.parse_args()
    train(args)