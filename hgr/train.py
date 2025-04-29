import os
import wandb
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim import AdamW

from landmark import MediapipeHandsLandmarks, MODEL_CONFIG
from dataset import HagridDataset, collate_filter_none
from evaluate import calculate_avg_distance, calculate_pck, evaluate_model
from torch_utils import get_device, optimize_for_device
from utils import save_checkpoint, set_seed

def main(args):
	# --- Setup --- 
	set_seed(args.seed)
	device = get_device()
	optimize_for_device(device)
	print(f"Using device: {device}")

	start_epoch = 0
	best_val_loss = float('inf')

	if args.wandb:
		wandb.init(project=args.project, name=args.run, config=vars(args))
		print(f"Wandb logging enabled for project '{args.project}', run '{args.run}'")

	# --- Data Loading ---
	print("Loading datasets...")
	train_dataset = HagridDataset(
		root=args.data_dir, 
		split="train", 
		augment=args.augment
	)
	val_dataset = HagridDataset(root=args.data_dir, split="val")
	test_dataset = HagridDataset(root=args.data_dir, split="test")

	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size, 
		shuffle=True, 
		num_workers=args.num_workers,
		collate_fn=collate_filter_none,
		pin_memory=True if device.type != 'cpu' else False
	)
	val_loader = DataLoader(
		val_dataset, 
		batch_size=args.batch_size, 
		shuffle=False, 
		num_workers=args.num_workers,
		collate_fn=collate_filter_none,
		pin_memory=True if device.type != 'cpu' else False
	)
	test_loader = DataLoader(
		test_dataset, 
		batch_size=args.batch_size, 
		shuffle=False, 
		num_workers=args.num_workers,
		collate_fn=collate_filter_none,
		pin_memory=True if device.type != 'cpu' else False
	)
	print(f"Train samples: {len(train_dataset)}")
	print(f"Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

	# --- Model, Loss, Optimizer --- 
	print("Initializing model...")
	model = MediapipeHandsLandmarks(config=MODEL_CONFIG).to(device)

	if args.wandb:
		wandb.watch(model, log=None)

	# loss_fn = nn.MSELoss()
	loss_fn = nn.SmoothL1Loss()
	if args.optimizer.lower() == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.lr)
		print("Using Adam optimizer.")
	elif args.optimizer.lower() == 'adamw':
		optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		print("Using AdamW optimizer.")
	else:
		raise ValueError(f"Unsupported optimizer: {args.optimizer}. Choose 'adam' or 'adamw'.")
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
	print(f"Model parameters: {model.count_parameters():,}")

	# --- Load Initial Weights from Checkpoint (if specified) ---
	if args.checkpoint:
		if os.path.isfile(args.checkpoint):
			print(f"Loading initial model weights from checkpoint: {args.checkpoint}")
			checkpoint = torch.load(args.checkpoint, map_location=device)
			if 'model_state_dict' in checkpoint:
				model.load_state_dict(checkpoint['model_state_dict'])
				print("Model weights loaded successfully.")
			else:
				print(f"Warning: 'model_state_dict' not found in checkpoint {args.checkpoint}. Weights not loaded.")
		else:
			print(f"Warning: Checkpoint file not found at {args.checkpoint}. Starting with randomly initialized weights.")

	# --- Training Loop --- 
	print(f"Starting training for {args.epochs} epochs...")
	best_val_loss = float('inf')
	best_train_loss = float('inf')
	scale_factor = 256.0
	pck_threshold = 20.0

	for epoch in range(start_epoch, args.epochs):
		model.train()
		train_loss_epoch = 0.0
		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

		for batch in pbar:
			if batch is None: 
				continue
				
			images, landmarks_gt = batch
			images = images.to(device)
			landmarks_gt = landmarks_gt.to(device)

			landmarks_pred = model(images)
			loss = loss_fn(landmarks_pred, landmarks_gt)
			optimizer.zero_grad()
			loss.backward()

			# Gradient Clipping
			if args.clip_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

			optimizer.step()

			train_loss_epoch += loss.item() * images.size(0)
			pbar.set_postfix(loss=loss.item())

		avg_train_loss = train_loss_epoch / len(train_dataset)

		# --- Val set evaluation --- 
		print(f"Epoch {epoch+1} completed. Evaluating validation set...")
		val_loss, val_avg_dist, val_pck, _, _ = evaluate_model(
			model, val_loader, device, loss_fn, scale_factor, pck_threshold
		)

		print(f"Epoch {epoch+1}: " 
			  f"Train Loss: {avg_train_loss:.4f} | " 
			  f"Val Loss: {val_loss:.4f}, Val AvgDist: {val_avg_dist:.2f}px, Val PCK@{pck_threshold}px: {val_pck:.4f}")

		# Log to Wandb
		if args.wandb:
			current_lr = optimizer.param_groups[0]['lr']
			wandb.log({
				"epoch": epoch + 1,
				"train/loss": avg_train_loss,
				"val/loss": val_loss,
				"val/avg_distance_px": val_avg_dist,
				"val/pck": val_pck,
				"learning_rate": current_lr
			})

		# --- Checkpoint ---
		# Save best model based on validation loss
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			state = {
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'val_loss': best_val_loss
			}
			save_checkpoint(state, "best_val.pth", args.project, args.run)
			print(f"Saved best validation model checkpoint (Loss: {best_val_loss:.4f}) at epoch {epoch+1}")

		# Save best model based on training loss
		if avg_train_loss < best_train_loss:
			best_train_loss = avg_train_loss
			state = {
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'train_loss': best_train_loss
			}
			save_checkpoint(state, "best_train.pth", args.project, args.run)
			print(f"Saved best training model checkpoint (Loss: {best_train_loss:.4f}) at epoch {epoch+1}")

		scheduler.step()

	# --- Test Set Evaluation --- 
	print("\nTraining complete. Evaluating test set...")

	final_test_loss, final_test_avg_dist, final_test_pck, _, _ = evaluate_model(
		model, test_loader, device, loss_fn, scale_factor, pck_threshold
	)

	print("--- Final Test Set Results ---")
	print(f"SmoothL1 Loss: {final_test_loss:.4f}")
	print(f"Average Distance: {final_test_avg_dist:.2f} pixels")
	print(f"PCK@{pck_threshold}px: {final_test_pck:.4f}")

	if args.wandb:
		wandb.log({
			"test/loss": final_test_loss,
			"test/avg_distance_px": final_test_avg_dist,
			"test/pck": final_test_pck,
		})
		wandb.finish()

	print("Evaluation complete.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train Mediapipe Hands Landmark Model")
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	default_project = f"project_{timestamp}"
	default_run = f"run_{timestamp}"

	# Data & Paths
	parser.add_argument("--data_dir", type=str, default="data/hagrid_small", help="Root directory of the dataset")
	
	# Training Hyperparameters
	parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
	parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
	parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument("--augment", action='store_true', help="Enable data augmentation for the training set")
	parser.add_argument("--clip_grad_norm", type=float, default=5.0, help="Max norm for gradient clipping (0 to disable)")
	parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR scheduler")
	parser.add_argument("--optimizer", type=str, default='adamw', choices=['adam', 'adamw'], help="Optimizer to use ('adam' or 'adamw')")
	parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for AdamW optimizer")

	# Wandb Logging
	parser.add_argument("--wandb", action='store_true', help="Enable Weights & Biases logging")
	parser.add_argument("--project", type=str, default=default_project, help="Wandb project name")
	parser.add_argument("--run", type=str, default=default_run, help="Wandb run name")

	# Checkpointing
	parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to initialize model weights from")

	args = parser.parse_args()
	main(args)
