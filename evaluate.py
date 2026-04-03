import torch
import argparse
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryConfusionMatrix
from tqdm import tqdm
import torch.nn.functional as F

from data import HistopathDataset, get_transforms, get_transforms_dinov2
import pandas as pd
import numpy as np
from models import HDFFModule, HistopathLightningModule, ContrastiveHDFFModule, Dinov2Module

def generate_kaggle_submission(model_module, dataset_test, threshold=0.5, batch_size=64, device="cuda", use_tta=True, output_file="submission.csv"):
    """
    Génère un fichier CSV pour Kaggle : ID, Pred
    """
    model_module.to(device)
    model_module.eval()
    
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        num_workers=8,
        shuffle=False, # CRITIQUE : Ne pas mélanger pour garder l'ordre des IDs
        pin_memory=(device == "cuda")
    )
    
    all_preds = []
    all_ids = []
    
    print(f"Post-processing en cours... (TTA: {use_tta})")
    
    with torch.no_grad():
        for images, _, indices in tqdm(test_loader, desc="Kaggle Inference"):
            images = images.to(device)
            
            if use_tta:
                # TTA simple : Original + Flips
                img_h = torch.flip(images, dims=[3])
                img_v = torch.flip(images, dims=[2])
                img_hv = torch.flip(images, dims=[2, 3])
                
                probs_sum = 0
                for v in [images, img_h, img_v, img_hv]:
                    logits = model_module(v).view(-1)
                    probs_sum += torch.sigmoid(logits)
                probs = probs_sum / 4
            else:
                logits = model_module(images).view(-1)
                probs = torch.sigmoid(logits)
            
            # Conversion en labels binaires (0 ou 1) selon le seuil
            preds = (probs >= threshold).int().cpu().numpy()
            
            all_preds.extend(preds)
            all_ids.extend(indices.numpy()) # On récupère les IDs du dataset

    # Création du DataFrame et export
    df = pd.DataFrame({
        "ID": all_ids,
        "Pred": all_preds
    })
    
    # Tri par ID pour être propre (souvent requis par Kaggle)
    df = df.sort_values(by="ID")
    df.to_csv(output_file, index=False)
    print(f"✅ Soumission sauvegardée sous : {output_file}")

def generate_ensemble_submission(model_paths, dataset_test, threshold=0.570, batch_size=64, device="cuda", use_tta=True, output_file="submission_ensemble.csv"):
    """
    Génère une soumission en moyennant les prédictions de 3 modèles avec POIDS OPTIMISÉS.
    Configuration : ConvNeXt (1.0), DINOv2 (2.0), HDFF (0.5)
    """
    # 1. Configuration des poids optimisés
    weights = [1.0, 1.0, 1.0] # Ordre : ConvNeXt, DINOv2, HDFF
    sum_weights = sum(weights)
    
    print(f"📦 Chargement de l'ensemble pondéré ({weights})...")
    models = []
    
    configs = [
        (HistopathLightningModule, model_paths['convnext']),
        (Dinov2Module, model_paths['dinov2']),
        (HDFFModule, model_paths['hdff'])
    ]

    for module_class, path in configs:
        m = module_class.load_from_checkpoint(path, map_location=device)
        m.to(device)
        m.eval()
        models.append(m)

    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        num_workers=8,
        shuffle=False, 
        pin_memory=(device == "cuda")
    )
    
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, _, indices in tqdm(test_loader, desc="Ensemble Inference"):
            images = images.to(device)
            weighted_batch_probs = torch.zeros(images.size(0), device=device)

            # Vues pour TTA
            views = [images]
            if use_tta:
                views += [
                    torch.flip(images, dims=[3]), 
                    torch.flip(images, dims=[2]), 
                    torch.flip(images, dims=[2, 3])
                ]

            # Inférence moyennée et pondérée
            for idx, model in enumerate(models):
                model_view_probs = 0
                for v in views:
                    logits = model(v).view(-1)
                    model_view_probs += torch.sigmoid(logits)
                
                # Moyenne TTA pour ce modèle spécifique
                model_avg_prob = model_view_probs / len(views)
                
                # Application du poids du modèle
                weighted_batch_probs += (model_avg_prob * weights[idx])
            
            # Moyenne finale de l'ensemble pondéré
            final_batch_probs = weighted_batch_probs / sum_weights
            
            all_probs.extend(final_batch_probs.cpu().numpy())
            all_ids.extend(indices.numpy())

    # Conversion en labels binaires selon le SEUIL OPTIMISÉ (0.570)
    all_preds = (np.array(all_probs) >= threshold).astype(int)

    df = pd.DataFrame({"ID": all_ids, "Pred": all_preds})
    df = df.sort_values(by="ID")
    df.to_csv(output_file, index=False)
    print(f"✅ Ensemble submission optimisée (T={threshold}) sauvegardée : {output_file}")

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

                # views_rot = [torch.rot90(v, k=1, dims=[2, 3]) for v in views]
                # all_views = views + views_rot
                all_views = views

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
    parser.add_argument("--checkpoint", type=str, default = None, help="Path to .ckpt")
    parser.add_argument("--test_path", type=str, default="data/test.h5", help="Path to test H5 file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for metrics")
    parser.add_argument("--use_tta", action="store_true", help="Use Test-Time Augmentation (TTA) for evaluation")
    parser.add_argument("--module_name", type=str, default="hdff", choices=["hdff","hdff_scl", "convnextv2", "dinov2"], help="Which model module to load from checkpoint")
    parser.add_argument("--kaggle_submission", action="store_true", help="Whether to generate a Kaggle submission CSV instead of just evaluating")
    parser.add_argument("--ensemble", action="store_true", help="Activer le mode Ensembling avec les 3 modèles prédéfinis")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n--- Initializing Evaluation ---")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.test_path}")

    dataset_test = HistopathDataset(args.test_path, transforms=get_transforms_dinov2(mode='val'))
    if args.ensemble:
        print("⚠️  Mode Ensembling activé : Les prédictions seront moyennées sur les 3 modèles suivants :")
        print("   - HDFF")
        print("   - ConvNextV2")
        print("   - Dinov2")
        model_paths = {
            'convnext': "models/final_models/convnextv2_tiny.ckpt",
            'dinov2': "models/final_models/dinov2_2-model-epoch=15.ckpt",
            'hdff': "models/final_models/hdff_model_1.ckpt"
        }
        generate_ensemble_submission(
            model_paths, 
            dataset_test, 
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=device,
            use_tta=args.use_tta
        )
    else:
        # Load Model
        try:
            if args.module_name == "hdff":
                model_module = HDFFModule.load_from_checkpoint(args.checkpoint, map_location=device)
            elif args.module_name == "convnextv2":
                model_module = HistopathLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
            elif args.module_name == "hdff_scl":
                model_module = ContrastiveHDFFModule.load_from_checkpoint(args.checkpoint, map_location=device)
            elif args.module_name == "dinov2":
                model_module = Dinov2Module.load_from_checkpoint(args.checkpoint, map_location=device)
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return

        if args.kaggle_submission:
            output_csv = "submission_kaggle.csv"

            # Génération
            generate_kaggle_submission(
                model_module, 
                dataset_test, 
                threshold=args.threshold, 
                batch_size=args.batch_size, 
                device=device, 
                use_tta=args.use_tta,
                output_file=output_csv
            )
        else:
            res = evaluate_on_test_set(model_module, dataset_test, batch_size=args.batch_size, device=device, threshold=args.threshold, use_tta=args.use_tta)
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