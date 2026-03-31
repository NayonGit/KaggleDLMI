import torch
import argparse
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryConfusionMatrix
from tqdm import tqdm
import torch.nn.functional as F

from data import HistopathDataset, get_transforms
from models import HDFFModule, HistopathLightningModule

def evaluate_on_test_set(model_module, dataset_test, threshold=0.5,batch_size=64, device="cuda", use_tta=True):
    """
    Calcule les métriques de classification sur le set de test.
    """
    model_module.to(device)
    model_module.eval()
    
    # Initialisation des métriques (TorchMetrics gère l'accumulation proprement)
    metrics = {
        "acc": BinaryAccuracy(threshold=threshold).to(device),
        "auc": BinaryAUROC().to(device),
        "f1": BinaryF1Score(threshold=threshold).to(device),
        "conf_mat": BinaryConfusionMatrix(threshold=threshold).to(device)
    }
    
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        num_workers=4,
        pin_memory=(device == "cuda")
    )
    
    print(f"🚀 Inférence en cours sur {len(dataset_test)} images...")
    
    with torch.no_grad():
        for images, targets, _ in tqdm(test_loader, desc="Evaluation"):
            images = images.to(device)
            targets = targets.to(device).float().squeeze()   # On s'assure que c'est du float pour les métriques
            if use_tta:
                img_h = torch.flip(images, dims=[3])
                img_v = torch.flip(images, dims=[2])
                img_hv = torch.flip(images, dims=[2, 3])
                views = [images, img_h, img_v, img_hv]

                views_rot = [torch.rot90(v, k=1, dims=[2, 3]) for v in views]
                all_views = views + views_rot
                probs_sum = 0

                for v in all_views:
                    logits = model_module(v).view(-1)
                    probs_sum += torch.sigmoid(logits)
                probs = probs_sum / len(all_views)

            else: #forward pass
                logits = model_module(images).view(-1)
                probs = torch.sigmoid(logits)
            
            
            # Update des métriques
            for metric in metrics.values():
                metric.update(probs, targets)
            
    # Calcul final
    results = {name: metric.compute() for name, metric in metrics.items()}
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Histopath Classification model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt")
    parser.add_argument("--test_path", type=str, default="data/test.h5", help="Path to test H5 file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for metrics")
    parser.add_argument("--use_tta", action="store_true", help="Use Test-Time Augmentation (TTA) for evaluation")
    parser.add_argument("--module_name", type=str, default="hdff", choices=["hdff", "convnextv2"], help="Which model module to load from checkpoint")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n--- Initializing Evaluation ---")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.test_path}")

    # Dataset (on utilise les transforms de validation : Resize/Normalisation uniquement)
    dataset_test = HistopathDataset(args.test_path, transforms=get_transforms(mode='val'))

    # Load Model
    try:
        if args.module_name == "hdff":
            model_module = HDFFModule.load_from_checkpoint(args.checkpoint, map_location=device)
        elif args.module_name == "convnextv2":
            model_module = HistopathLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # Run Evaluation
    res = evaluate_on_test_set(model_module, dataset_test, batch_size=args.batch_size, device=device, threshold=args.threshold)

    # Affichage des résultats
    cm = res['conf_mat'].cpu().numpy()
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*45)
    print(f" 🔬 TEST RESULTS SUMMARY ")
    print("="*45)
    print(f" Accuracy  : {res['acc']:.4f}")
    print(f" AUROC     : {res['auc']:.4f}")
    print(f" F1-Score  : {res['f1']:.4f}")
    print("-" * 45)
    print(f" Confusion Matrix :")
    print(f"   [ TN: {int(tn):<5} | FP: {int(fp):<5} ]")
    print(f"   [ FN: {int(fn):<5} | TP: {int(tp):<5} ]")
    print("-" * 45)
    
    # Calcul de la Sensibilité (Rappel) et Spécificité
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f" Sensitivity (Recall) : {sensitivity:.4f}")
    print(f" Specificity          : {specificity:.4f}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()