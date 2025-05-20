# src/infer.py

import os
import csv
import numpy as np
import torch
from PIL import Image

from src.model import get_model
from src.utils import load_model, get_transform, normalize


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    internal_labels = [
        "Amandina","Arabia","Comtesse","Creme_brulee","Jelly_Black",
        "Jelly_Milk","Jelly_White","Noblesse","Noir_authentique",
        "Passion_au_lait","Stracciatella","Tentation_noir","Triangolo",
    ]

    model = get_model(len(internal_labels)+1)
    model = load_model(model, args.model_path, device)

    submission_labels = args.label_names.split(',')
    tf = get_transform(train=False)

    # Preparing the CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id'] + submission_labels)

        for fname in sorted(os.listdir(args.test_folder)):
            if not fname.lower().endswith(('.JPG','.jpg','.jpeg','.png','.PNG')):
                continue
            img_id = os.path.splitext(fname)[0].lstrip('L')
            pil = Image.open(os.path.join(args.test_folder, fname)).convert("RGB")
            
            # Applying the same transformations as train time
            img_tensor, _ = tf(pil, {})
            img_tensor = img_tensor.to(device)

            # Inference
            with torch.no_grad():
                preds = model([img_tensor])[0]

            # Only keeping detections above confidence threshold
            scores = preds['scores'].cpu().numpy()
            lbls   = preds['labels'].cpu().numpy()
            keep   = scores >= args.conf_thresh
            kept   = lbls[keep]

            counts_int = np.bincount(kept, minlength=len(internal_labels)+1)[1:]
            
            # Stripping accents and spaces so that "Crème brulée" becomes "Creme_brulee"
            idx_map = []
            for name in submission_labels:
                norm = normalize(name)
                idx_map.append(internal_labels.index(norm))

            counts = [int(counts_int[i]) for i in idx_map]
            writer.writerow([img_id] + counts)

    print(f"Saved predictions to {args.output_csv}")
