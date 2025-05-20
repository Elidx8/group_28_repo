# src/train.py

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from sklearn.model_selection import KFold

from src.dataset import ChocolateCocoDataset
from src.model import get_model
from src.utils import (
    get_transform, collate_fn,
    save_model, visualize_predictions,
    compute_counts_from_preds, image_f1
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss_classifier = loss_dict['loss_classifier']
        loss_box_reg    = loss_dict['loss_box_reg']
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(
                f"Epoch {epoch} [{i}/{len(data_loader)}]  "
                f"cls={loss_classifier.item():.4f}  "
                f"box={loss_box_reg.item():.4f}  "
                f"total={losses.item():.4f}"
            )
            # quick IoU check
            model.eval()
            with torch.no_grad():
                preds = model(images)
            ious = []
            for p, t in zip(preds, targets):
                if len(p["boxes"]) and len(t["boxes"]):
                    m = box_iou(p["boxes"], t["boxes"])
                    best_pred_iou, _ = m.max(dim=1)
                    ious.append(best_pred_iou.mean().item())
            avg_iou = sum(ious)/len(ious) if ious else 0.0
            print(f"    avg batch IoU: {avg_iou:.3f}")
            model.train()

def run_training(args):
    # reproducibility
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    base_imgs = args.train_imgs
    base_json = args.coco_json
    idx_dataset = ChocolateCocoDataset(base_imgs, base_json, transforms=None)
    labels = idx_dataset.labels

    # Cross Validation
    if args.k_folds > 0 and not args.skip_cv:
        
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(idx_dataset), 1):
            print(f"\n=== Fold {fold}/{args.k_folds} ===")
            
            # Creating output directories for each fold
            run_dir    = os.path.join(args.output_root, f"{args.run_name}_fold{fold}")
            models_dir = os.path.join(run_dir, "models")
            vis_dir    = os.path.join(run_dir, "visuals")
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(vis_dir,    exist_ok=True)
            
            # Train and validation datasets
            train_full = ChocolateCocoDataset(
                base_imgs, base_json, transforms=get_transform(train=True)
            )
            val_full   = ChocolateCocoDataset(
                base_imgs, base_json, transforms=get_transform(train=False)
            )
            
            train_ds = Subset(train_full, train_idx)
            val_ds   = Subset(val_full,   val_idx)
            
            # Data loaders
            g = torch.Generator() 
            g.manual_seed(args.seed)
            
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
                collate_fn=collate_fn,
                worker_init_fn=seed_worker,
                generator=g
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                collate_fn=collate_fn,
                worker_init_fn=seed_worker,
                generator=g
            )
            
            # Model, optimizer, scheduler
            num_classes = len(labels) + 1
            model = get_model(num_classes).to(device)
            print(f"  → Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.6f} M")
            
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9, weight_decay=5e-4)
            
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.step_size, gamma=args.gamma
            )
            
            best_val_f1, no_improve = 0.0, 0

            # Epochs loop
            for epoch in range(1, args.epochs+1):
                train_one_epoch(model, optimizer, train_loader, device, epoch)
                lr_scheduler.step()
                
                # Validation F1 score 
                model.eval()
                all_f1 = []
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    
                    with torch.no_grad():
                        batch_preds = model(images)
                        
                    for t, p in zip(targets, batch_preds):
                        # ground-truth counts:
                        true_counts = t["labels"].cpu().numpy()
                        y_true = np.bincount(true_counts, minlength=num_classes+1)[1:]
                        # predicted counts:
                        y_pred = compute_counts_from_preds(p, args.conf_thresh_score, num_classes)
                        all_f1.append(image_f1(y_true, y_pred))
                        
                avg_val_f1 = float(np.mean(all_f1))
                print(f" → Val F1: {avg_val_f1:.4f}")

                # Early stopping based on F1 score
                if avg_val_f1 > best_val_f1:
                    best_val_f1 = avg_val_f1
                    no_improve = 0
                    save_model(model, os.path.join(models_dir, "best.pth"))
                else:
                    no_improve += 1
                    
                if no_improve >= args.patience:
                    print(f"No F1 improvement for {args.patience} epochs, stopping.")
                    break
                    
                model.train()
                
                # Save and visualize
                visualize_predictions(
                    model, 
                    val_ds.dataset, 
                    labels, device,
                    epoch, vis_dir,
                    n_samples=args.visual_samples,
                    conf_thresh=args.conf_thresh_vis
                )
                
            save_model(model, os.path.join(models_dir, "last.pth"))

        print("All folds complete.\nTraining complete.")
        
    else:
        print("Skipping cross-validation; proceeding straight to final retrain…")

    # ──────────────────────────────────────────────────────────────────────────
    # Final retrain on ALL data
    print("\n▶︎ Retraining on full dataset for final model…")
    
    final_dir = os.path.join(args.output_root, f"{args.run_name}_final")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(os.path.join(final_dir, "visuals"), exist_ok=True)

    # Full dataset and loader
    full_ds = ChocolateCocoDataset(
        base_imgs, base_json, transforms=get_transform(train=True)
    )
    
    g_full = torch.Generator(); g_full.manual_seed(args.seed)

    full_loader = DataLoader(full_ds, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             collate_fn=collate_fn,
                             worker_init_fn=seed_worker,
                             generator=g_full)

    # Validation held-out dataset and loader
    full_val_ds = ChocolateCocoDataset(
        base_imgs, base_json, transforms=get_transform(train=False)
    )
    
    val_loader_full = DataLoader(full_val_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn,
                                 worker_init_fn=seed_worker,
                                 generator=g_full)

    # Building a fresh model and optimizer/scheduler
    num_classes = len(labels) + 1
    final_model = get_model(num_classes).to(device)
    final_optimizer = torch.optim.SGD(
        final_model.parameters(),
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    final_scheduler = torch.optim.lr_scheduler.StepLR(
        final_optimizer, step_size=args.step_size, gamma=args.gamma
    )
    
    best_full_f1 = 0.0
    no_imp_full  = 0

    # Train for final_epochs
    for epoch in range(1, args.final_epochs+1):
        train_one_epoch(
            final_model, 
            final_optimizer, 
            full_loader, 
            device, 
            epoch
        )
        final_scheduler.step()

        # Computing F1 on full_val_ds
        final_model.eval()
        all_f1 = []
        for images, targets in val_loader_full:
            images = [img.to(device) for img in images]
            with torch.no_grad():
                preds = final_model(images)
            for t, p in zip(targets, preds):
                y_true = np.bincount(t["labels"].cpu().numpy(),
                                    minlength=num_classes+1)[1:]
                y_pred = compute_counts_from_preds(p, args.conf_thresh_score, num_classes)
                all_f1.append(image_f1(y_true, y_pred))
                
        avg_full_f1 = float(np.mean(all_f1))
        print(f" → Full‐data Val F1: {avg_full_f1:.4f}")

        # Early stop / saving best_full.pth
        if avg_full_f1 > best_full_f1:
            best_full_f1, no_imp_full = avg_full_f1, 0
            save_model(final_model, os.path.join(final_dir, "best_full.pth"))
            
        else:
            no_imp_full += 1
            
        if no_imp_full >= args.patience:
            print(f"No full‐data F1 improvement for {args.patience} epochs, stopping final retrain.")
            break
            
        visualize_predictions(
            final_model, 
            full_val_ds, 
            labels, 
            device,
            epoch, os.path.join(final_dir, "visuals"),
            n_samples=args.visual_samples,
            conf_thresh=args.conf_thresh_vis
        )
        final_model.train()

    # Saving last model
    save_model(final_model, os.path.join(final_dir, "last.pth"))
