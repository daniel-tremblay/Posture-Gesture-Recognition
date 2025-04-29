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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import torchmetrics 

from gesture import GestureClassifier
from evaluate import evaluate_classifier
from dataset import HagridDataset, collate_filter_none
from torch_utils import get_device, optimize_for_device
from utils import save_checkpoint, set_seed

def main(args):
	# --- Setup ---
	set_seed(args.seed)
	device = get_device()
	optimize_for_device(device)
	print(f"Using device: {device}")

	start_epoch = 0
	best_val_metric = 0.0

	if args.wandb:
		wandb.init(project=args.project, name=args.run, config=vars(args))
		print(f"Wandb logging enabled for project '{wandb.config.project}', run '{wandb.config.run}'")

	# --- Data Loading ---
	print("Loading datasets...")

	train_dataset = HagridDataset(root=args.data_dir, split="train", augment=args.augment, crop_padding_factor=0.2)
	val_dataset = HagridDataset(root=args.data_dir, split="val", augment=False, crop_padding_factor=0.2)
	test_dataset = HagridDataset(root=args.data_dir, split="test", augment=False, crop_padding_factor=0.2)

	if train_dataset.num_classes == 0:
		print("Error: Number of classes detected is 0. Check dataset.")
		return
	num_classes = train_dataset.num_classes
	print(f"Number of gesture classes: {num_classes}")

	train_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=args.num_workers, collate_fn=collate_filter_none, pin_memory=True if device.type != 'cpu' else False
	)
	val_loader = DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, collate_fn=collate_filter_none, pin_memory=True if device.type != 'cpu' else False
	)
	test_loader = DataLoader(
		test_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, collate_fn=collate_filter_none, pin_memory=True if device.type != 'cpu' else False
	)
	print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

	# --- Model, Loss, Optimizer ---
	print("Initializing Gesture Classifier model...")
	model = GestureClassifier(num_classes=num_classes).to(device)
	print(f"Initial trainable parameters (Head Only): {model.count_parameters(trainable_only=True):,}")

	if args.wandb:
		wandb.watch(model, log=None)

	criterion = nn.CrossEntropyLoss()

	optimizer = AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=args.lr,
		weight_decay=args.weight_decay
	)
	print(f"Using AdamW optimizer with LR={args.lr}, WD={args.weight_decay}")
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

	# --- Training Loop ---
	# Phase 1: Head Training
	# Phase 2: Fine-Tuning

	print(f"\nStarting training process ({args.head_epochs} head epochs + {args.epochs - args.head_epochs} fine-tuning epochs)")
	current_phase = 1

	print(f"\n--- Starting Phase 1: Head Training ---")
	for epoch in range(start_epoch, args.epochs):

		if current_phase == 1 and epoch >= args.head_epochs:
			print(f"\n--- Epoch {epoch+1}: Switching to Phase 2: Fine-Tuning ---")
			current_phase = 2
			model.unfreeze_backbone(stages_to_unfreeze=args.unfreeze_stages) # Unfreeze layers

			print("Re-initializing optimizer for fine-tuning...")
			optimizer = AdamW(
				filter(lambda p: p.requires_grad, model.parameters()),
				lr=args.finetune_lr,
				weight_decay=args.weight_decay
			)
			print(f"Fine-tuning trainable parameters: {model.count_parameters(trainable_only=True):,}")
	
			remaining_epochs = args.epochs - epoch
			scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=args.eta_min)
			print(f"Reset CosineAnnealingLR scheduler for remaining {remaining_epochs} epochs.")

		# --- Training Epoch ---
		model.train()
		train_loss_epoch = 0.0
		train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

		phase_str = "Head" if current_phase == 1 else "Finetune"
		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [{phase_str} Training]")

		for batch in pbar:
			if batch is None or len(batch) != 3 or batch[0] is None or batch[2] is None: continue
			images, _, labels = batch
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)
			optimizer.zero_grad()
			loss.backward()

			if args.clip_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(
					filter(lambda p: p.requires_grad, model.parameters()),
					args.clip_grad_norm
				)
			optimizer.step()

			train_loss_epoch += loss.item() * images.size(0)
			train_acc_metric.update(outputs, labels)
			pbar.set_postfix(loss=loss.item())

		avg_train_loss = train_loss_epoch / len(train_dataset) if len(train_dataset) > 0 else 0
		train_accuracy = train_acc_metric.compute().item()
		train_acc_metric.reset()

		val_loss, val_accuracy, val_f1 = evaluate_classifier(
			model, val_loader, criterion, device, num_classes, prefix="val"
		)

		print(
			f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
			f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}"
		)

		scheduler.step()

		# --- Logging ---
		if args.wandb:
			wandb.log({
				"epoch": epoch + 1,
				"train/loss": avg_train_loss,
				"train/accuracy": train_accuracy,
				"val/loss": val_loss,
				"val/accuracy": val_accuracy,
				"val/f1_score": val_f1,
				"learning_rate": optimizer.param_groups[0]['lr'],
				"phase": current_phase
			})

		current_val_metric = val_f1
		if current_val_metric > best_val_metric:
			best_val_metric = current_val_metric
			state = {
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'val_accuracy': val_accuracy,
				'val_f1': val_f1,
				'num_classes': num_classes,
				'scheduler_state_dict': scheduler.state_dict()
			}
			save_dir_project = wandb.config.project if args.wandb else args.project
			save_dir_run = wandb.config.run if args.wandb else args.run
			save_checkpoint(state, "best_val_f1.pth", save_dir_project, save_dir_run)
			print(f"Saved best validation model checkpoint (F1: {best_val_metric:.4f}, Accuracy: {val_accuracy:.4f}) at epoch {epoch+1}")

	# --- Test Set Evaluation ---
	print("\nTraining complete. Evaluating test set with best validation accuracy model...")
	try:
		save_dir_project = wandb.config.project if args.wandb else args.project
		save_dir_run = wandb.config.run if args.wandb else args.run
		best_model = os.path.join("checkpoints", save_dir_project, save_dir_run, "best_val_f1.pth")
		checkpoint = torch.load(best_model, map_location=device)
		ckpt_num_classes = checkpoint.get('num_classes', num_classes)
		if ckpt_num_classes != num_classes:
			print(f"Warning: Checkpoint num_classes ({ckpt_num_classes}) differs from dataset ({num_classes}). Using value from dataset.")
		
		model = GestureClassifier(
			num_classes=num_classes, 
			pretrained=False
		).to(device)
		model.load_state_dict(checkpoint['model_state_dict'])
		print(f"Loaded best model from epoch {checkpoint.get('epoch', 'N/A')} with validation accuracy {checkpoint.get('val_accuracy', 0):.4f}")
	except FileNotFoundError:
	     print(f"Error: Best checkpoint not found at {best_model}. Evaluating model from last epoch.")
	except Exception as e:
		print(f"Error loading best checkpoint: {e}. Evaluating model from last epoch.")

	test_loss, test_accuracy, test_f1 = evaluate_classifier(
		model, test_loader, criterion, device, num_classes, prefix="test"
	)

	print("\n--- Test Set Results ---")
	print(f"  Test Loss: {test_loss:.4f}")
	print(f"  Test Accuracy: {test_accuracy:.4f}")
	print(f"  Test F1-Score (Macro): {test_f1:.4f}")

	if args.wandb:
		wandb.log({
			"test/loss": test_loss,
			"test/accuracy": test_accuracy,
			"test/f1_score": test_f1,
		})
		wandb.finish()

	print("\nTraining complete.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train Hand Gesture Classifier using Transfer Learning")

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	default_project = f"project_{timestamp}"
	default_run = f"run_{timestamp}"

	parser.add_argument("--data_dir", type=str, default="data/hagrid_small", help="Root directory of the dataset")
	parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs (head + fine-tune)")
	parser.add_argument("--head_epochs", type=int, default=10, help="Number of epochs to train only the classifier head")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
	parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (for head training)")
	parser.add_argument("--finetune_lr", type=float, default=1e-5, help="Learning rate for fine-tuning phase (after unfreezing)")
	parser.add_argument("--unfreeze_stages", type=int, default=1, help="Num ResNet stages to unfreeze from end for fine-tuning (-1 for all, 0 for none/head only, 1-4)")
	parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (esp. for AdamW)")
	parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR scheduler")
	parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping (0 to disable)")
	parser.add_argument("--augment", action='store_true', help="Enable data augmentation for the training set")
	parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument("--wandb", action='store_true', help="Enable Weights & Biases logging")
	parser.add_argument("--project", type=str, default=default_project, help="Wandb project name")
	parser.add_argument("--run", type=str, default=default_run, help="Wandb run name")

	args = parser.parse_args()

	if args.head_epochs >= args.epochs:
		print("Warning: head_epochs >= total epochs. Fine-tuning phase will not run.")
		args.head_epochs = args.epochs

	main(args)