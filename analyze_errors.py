import torch
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import HistopathDataset, get_transforms_dinov2
from models import Dinov2Module

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" usando {device} para el análisis...")

    # 1. Chargement du Dataset (Mode Validation pour avoir les labels)
    transform = get_transforms_dinov2(mode='val')
    dataset = HistopathDataset(args.data_path, transforms=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

    # 2. Chargement du Modèle
    model = Dinov2Module.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    results = []
    print("🚀 Inférence sur le set de validation...")
    
    with torch.no_grad():
        for x, y, ids in tqdm(loader):
            logits = model(x.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true = y.cpu().numpy().flatten()
            ids = ids.numpy()

            for i in range(len(y_true)):
                # Calcul de l'erreur absolue
                error = np.abs(y_true[i] - probs[i])
                results.append({
                    'id': int(ids[i]),
                    'label': int(y_true[i]),
                    'prob': float(probs[i]),
                    'error': float(error)
                })

    # 3. Analyse avec Pandas
    df = pd.DataFrame(results)
    df = df.sort_values(by='error', ascending=False)
    
    # Sauvegarde du CSV pour analyse approfondie (ex: par centre)
    df.to_csv("validation_errors_analysis.csv", index=False)
    print(f"✅ Analyse sauvegardée dans validation_errors_analysis.csv")

    # 4. Visualisation des 10 pires erreurs
    print("🖼️ Génération de la planche des pires erreurs...")
    top_errors = df.head(10)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(top_errors.iterrows()):
        # On recharge l'image brute pour affichage (sans normalisation)
        img_id = str(int(row['id']))
        group = dataset.hdf[img_id]
        img = np.array(group['img'], dtype=np.float32).transpose(1, 2, 0)
        if img.max() <= 1.01: img *= 255
        img = img.astype(np.uint8)

        axes[i].imshow(img)
        title = f"ID: {img_id}\nTrue: {int(row['label'])} | Prob: {row['prob']:.3f}"
        axes[i].set_title(title, color='red' if row['error'] > 0.5 else 'black')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("top_errors_visualization.png")
    print(f"✅ Planche visuelle sauvegardée : top_errors_visualization.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/val.h5")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    main(args)