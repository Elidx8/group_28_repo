# main.py

import os
import argparse

from src.train import run_training
from src.infer import run_inference

def parse_args():
    p = argparse.ArgumentParser(
        description="Train (unless --model_path is precised), then infer"
    )
    # ─── Training args ─────────────────────────────────────────────────────────
    p.add_argument("--train_imgs",  required=True)
    p.add_argument("--coco_json",   required=True)
    p.add_argument("--output_root", default="output")
    p.add_argument("--run_name",    required=True)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs",      type=int, default=300)
    p.add_argument("--lr",          type=float, default=0.005)
    p.add_argument("--step_size",   type=int, default=80)
    p.add_argument("--gamma",       type=float, default=0.1)
    p.add_argument("--visual_samples", type=int, default=4)
    p.add_argument("--conf_thresh_score", type=float, default=0.60)
    p.add_argument("--conf_thresh_vis",   type=float, default=0.50)
    p.add_argument("--patience",    type=int,   default=30)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--k_folds",     type=int,   default=5)
    p.add_argument("--final_epochs", type=int,  default=300)
    p.add_argument("--skip_cv",     action="store_true")

    # ─── Optional checkpoint that skips training ───────────────────────────────
    p.add_argument(
        "--model_path",
        default=None,
        help="If set, skip training and load this checkpoint for inference"
    )

    # ─── Inference args ────────────────────────────────────────────────────────
    p.add_argument("--test_folder", required=True)
    p.add_argument("--output_csv",  required=True)
    p.add_argument(
        "--label_names",
        required=True,
        help="Comma-separated list of class names in training order"
    )
    p.add_argument("--conf_thresh", type=float, default=0.50)

    args = p.parse_args()
    if args.final_epochs is None:
        args.final_epochs = args.epochs
    return args

def main():
    args = parse_args()

    # 1) TRAINING (skipped if --model_path provided)
    if args.model_path is None:
        print("▶︎ Starting training…")
        run_training(args)
        args.model_path = os.path.join(
            args.output_root,
            f"{args.run_name}_final",
            "best_full.pth"
        )
    else:
        print(f"▶︎ Skipping training; will load checkpoint {args.model_path!r}")

    # 2) INFERENCE
    print("\n▶︎ Starting inference…")
    run_inference(args)

if __name__ == "__main__":
    main()
