import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryConfusionMatrix
from tqdm import tqdm


from data import HistopathDataset, get_transforms_fms
from models import UNI2Module, Dinov2Module, PhikonV2Module, Virchow2Module, GigaPathModule

def generate_kaggle_submission(model_module, dataset_test, threshold=0.5, batch_size=64, device="cuda", use_tta=True, output_file="submission.csv"):
    """
    Generates a CSV file for Kaggle submission: ID, Pred
    """
    model_module.to(device)
    model_module.eval()
    
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        num_workers=8,
        shuffle=False,
        pin_memory=(device == "cuda")
    )
    
    all_preds = []
    all_ids = []
    
    print(f"Post-processing... (TTA: {use_tta})")
    
    with torch.no_grad():
        for images, _, indices in tqdm(test_loader, desc="Kaggle Inference"):
            images = images.to(device)
            
            if use_tta:
                # Standard TTA: Original + Flips
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
            
            preds = (probs >= threshold).int().cpu().numpy()
            
            all_preds.extend(preds)
            all_ids.extend(indices.numpy()) 

    df = pd.DataFrame({
        "ID": all_ids,
        "Pred": all_preds
    })
    
    df = df.sort_values(by="ID")
    df.to_csv(output_file, index=False)
    print(f"✅ Submission saved to: {output_file}")


def evaluate_on_test_set(model_module, dataset_test, threshold=0.5,batch_size=64, device="cuda", use_tta=True):
    """
    Calculates the classification metrics on the test set.
    """
    model_module.to(device)
    model_module.eval()
    
    metrics = {
        "acc": BinaryAccuracy(threshold=threshold).to(device),
        "auc": BinaryAUROC().to(device),
        "f1": BinaryF1Score(threshold=threshold).to(device),
        "conf_mat": BinaryConfusionMatrix(threshold=threshold).to(device)
    }
    
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        num_workers=8,
        pin_memory=(device == "cuda")
    )
    
    print(f"Inference on {len(dataset_test)} images...")
    
    with torch.no_grad():
        for images, targets, _ in tqdm(test_loader, desc="Evaluation"):
            images = images.to(device)
            targets = targets.to(device).float().squeeze()  
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
            
            
            for metric in metrics.values():
                metric.update(probs, targets)
            
    results = {name: metric.compute() for name, metric in metrics.items()}
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Histopath Classification model.")
    parser.add_argument("--checkpoint", type=str, default = None, help="Path to .ckpt")
    parser.add_argument("--test_path", type=str, default="data/test.h5", help="Path to test H5 file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for metrics")
    parser.add_argument("--use_tta", action="store_true", help="Use Test-Time Augmentation (TTA) for evaluation")
    parser.add_argument("--module_name", type=str, default="hdff", choices=["uni2","phikonv2","dinov2","gigapath","virchow2"], help="Which model module to load from checkpoint")
    parser.add_argument("--kaggle_submission", action="store_true", help="Whether to generate a Kaggle submission CSV instead of just evaluating")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n--- Initializing Evaluation ---")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.test_path}")

    dataset_test = HistopathDataset(args.test_path, transforms=get_transforms_fms(mode='val'))
    print(f"Test Dataset Loaded : {len(dataset_test)} images")

    # Load Model
    MODEL_REGISTRY = {
    "uni2": UNI2Module,
    "phikonv2": PhikonV2Module,
    "dinov2": Dinov2Module,
    "gigapath": GigaPathModule,
    "virchow2": Virchow2Module
    }

    if args.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {args.model_name} unknown. Choose among: {list(MODEL_REGISTRY.keys())}")
    print(f"Model Initialization : {args.model_name}")
    model_class = MODEL_REGISTRY[args.model_name]
    model_module = model_class(lr=args.lr)
    

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
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f" Sensitivity (Recall) : {sensitivity:.4f}")
        print(f" Specificity          : {specificity:.4f}")
        print("="*45 + "\n")

if __name__ == "__main__":
    main()