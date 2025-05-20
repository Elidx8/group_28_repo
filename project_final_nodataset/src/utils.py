# src/utils.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
import torch

import src.transforms as T 


import random
from PIL import Image, ImageDraw, ImageFont
import os



def compute_counts_from_preds(preds, conf_thresh, num_classes):
    """
    preds is a dict with keys "boxes","scores","labels"
    returns a length-C array of counts for this image
    """
    keep = preds["scores"] >= conf_thresh
    labels = preds["labels"][keep].cpu().numpy()
    counts = np.bincount(labels, minlength=num_classes+1)[1:]  # drop background
    return counts

def image_f1(y_true, y_pred):
    tp = np.sum(np.minimum(y_true, y_pred))
    # by the Kaggle definition  F1 = 2*TP / (2*TP + FPN)
    # and FPN = FP+FN = sum(y_true)+sum(y_pred) - 2*TP
    denom = np.sum(y_true) + np.sum(y_pred)
    return 2*tp/denom if denom>0 else 1.0

def collate_fn(batch):
    return tuple(zip(*batch))

def normalize(name):
    s = unicodedata.normalize('NFKD', name)
    s = s.encode('ascii','ignore').decode('ascii')
    return s.replace(' ', '_')

def get_transform(train):
    return T.Compose([
        T.Resize(min_size=800, max_size=1333),
        T.ToTensor(),
    ])

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize_predictions(model, dataset, class_names, device,
                          epoch, out_dir, n_samples=4, conf_thresh=0.25):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), n_samples)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        
        img_tensor, targets = dataset[idx]
        
        img_meta = dataset.coco.loadImgs(dataset.ids[idx])[0]
        img_name = img_meta["file_name"]
        ax.set_title(img_name, fontsize=10)
        ax.axis('off')

        img = img_tensor.permute(1,2,0).cpu().numpy()
        ax.imshow(img)

        with torch.no_grad():
            preds = model([img_tensor.to(device)])[0]

        # Draw ground truth boxes in green with labels
        for box, lbl in zip(targets["boxes"].cpu().numpy(),
                            targets["labels"].cpu().numpy()):
            x1,y1,x2,y2 = box
            ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1,
                                       edgecolor='lime', fill=False, lw=2))
            ax.text(
                x1, y1,
                f"{class_names[lbl-1]}",
                color='lime',
                backgroundcolor='black',
                fontsize=8,
                verticalalignment='bottom'  
            )

        # Draw prediction boxes in red with scores and labels
        for box, label, score in zip(
                preds["boxes"].cpu().numpy(),
                preds["labels"].cpu().numpy(),
                preds["scores"].cpu().numpy()):
            if score < conf_thresh:
                continue
            x1,y1,x2,y2 = box
            ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1,
                                       edgecolor='red', fill=False, lw=2))
            ax.text(
                x1, y2,
                f"{class_names[label-1]}:{score:.2f}",
                color='red',
                backgroundcolor='black',
                fontsize=8,
                verticalalignment='top'  
            )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch{epoch:03d}.png"), dpi=300)
    plt.close(fig)
    model.train()
    

def visualize_test_prediction(model, test_folder, transforms, conf_thresh=0.5, device='cpu'):
    # Class names from the dataset
    class_names = [
        "Amandina","Arabia","Comtesse","Creme_brulee","Jelly_Black",
        "Jelly_Milk","Jelly_White","Noblesse","Noir_authentique",
        "Passion_au_lait","Stracciatella","Tentation_noir","Triangolo",
    ]

    
    # Get random test image
    test_files = [f for f in os.listdir(test_folder) 
                  if f.lower().endswith(('.jpg','.jpeg','.png'))]
    test_img_path = os.path.join(test_folder, random.choice(test_files))
    
    # Load and transform image
    img = Image.open(test_img_path).convert("RGB")
    img_tensor, _ = transforms(img, {})
    
    # Run inference
    model.eval()
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])[0]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display the transformed image
    img_display = img_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_display)
    plt.axis('off')
    
    # Filter predictions by confidence
    keep = prediction['scores'] > conf_thresh
    boxes = prediction['boxes'][keep]
    labels = prediction['labels'][keep]
    scores = prediction['scores'][keep]
    
    # Draw predictions
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 
              'magenta', 'orange', 'lime', 'pink', 'teal', 'lavender', 'brown']
    
    for box, label, score in zip(boxes, labels, scores):
        # Get coordinates
        x1, y1, x2, y2 = box.cpu().numpy()
        label_idx = int(label) - 1  # Convert 1-based to 0-based indexing
        
        # Draw box
        color = colors[label_idx % len(colors)]
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, edgecolor=color, linewidth=2))
        
        # Draw label and score
        text = f"{class_names[label_idx]}: {score:.2f}"
        plt.text(x1, y1-5, text,
                color=color,
                backgroundcolor='black',
                fontsize=8,
                verticalalignment='bottom')
    
    plt.title(f'Predictions on {os.path.basename(test_img_path)}')
    plt.show()