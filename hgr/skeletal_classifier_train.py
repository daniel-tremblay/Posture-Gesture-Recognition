import os
import wandb
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from tqdm import tqdm

from dataset import HagridDataset
from utils import save_checkpoint, set_seed
from torch_utils import get_device, optimize_for_device
from skeletal_classifier import SkeletalGestureClassifier, collate_landmarks

CLASSES_TO_IGNORE = [
  "peace_inverted", 
	"mute", 
	"no_gesture", 
	"stop_inverted", 
	"stop",
  "three2", 
	"two_up_inverted", 
	"two_up", 
	"four", 
	"three"
]

def evaluate_skeletal_classifier(
	model: SkeletalGestureClassifier,
	dataloader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	num_classes: int,
	prefix: str = "val"
) -> tuple[float, float, float]:
	"""Evaluates the skeletal gesture classification model."""
	model.eval()
	total_loss = 0.0
	processed_samples = 0

	accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
	f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)

	with torch.no_grad():
		for batch in tqdm(dataloader, desc=f"Evaluation ({prefix})", leave=False):
			if batch is None:
				print(f"Warning: Skipping None batch during {prefix} evaluation.")
				continue

			landmarks, labels = batch
			landmarks = landmarks.to(device)
			labels = labels.to(device)

			outputs = model(landmarks)
			loss = criterion(outputs, labels)
			
			batch_size = landmarks.size(0)
			total_loss += loss.item() * batch_size
			processed_samples += batch_size

			accuracy_metric.update(outputs, labels)
			f1_metric.update(outputs, labels)

	if processed_samples == 0:
		print(f"Warning: No samples processed during {prefix} evaluation.")
		return 0.0, 0.0, 0.0

	avg_loss = total_loss / processed_samples
	accuracy = accuracy_metric.compute().item()
	f1_score = f1_metric.compute().item()

	accuracy_metric.reset()
	f1_metric.reset()

	return avg_loss, accuracy, f1_score

def main(args):
	set_seed(args.seed)
	device = get_device()
	optimize_for_device(device)
	print(f"Using device: {device}")

	start_epoch = 0
	best_val_f1 = 0.0

	if args.wandb:
		wandb.init(project=args.project, name=args.run, config=vars(args))
		print(f"Wandb logging enabled for project '{wandb.config.project}', run '{wandb.config.run}'")

	# --- Loading Dataset ---
	print("Loading datasets for skeletal classification...")
	train_dataset = HagridDataset(
		root=args.data_dir, split="train", augment=args.augment,
		ignore_classes=CLASSES_TO_IGNORE, crop_padding_factor=0.0
	)
	val_dataset = HagridDataset(
		root=args.data_dir, split="val", augment=False,
		ignore_classes=CLASSES_TO_IGNORE, crop_padding_factor=0.0
	)
	test_dataset = HagridDataset(
		root=args.data_dir, split="test", augment=False,
		ignore_classes=CLASSES_TO_IGNORE, crop_padding_factor=0.0
	)

	if train_dataset.num_classes == 0:
		print("Error: Number of classes detected is 0. Check dataset and ignored classes.")
		return
	num_classes = train_dataset.num_classes
	print(f"Number of gesture classes for skeletal model: {num_classes}")

	train_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=args.num_workers, collate_fn=collate_landmarks,
		pin_memory=True if device.type != 'cpu' else False
	)
	val_loader = DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, collate_fn=collate_landmarks,
		pin_memory=True if device.type != 'cpu' else False
	)
	test_loader = DataLoader(
		test_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, collate_fn=collate_landmarks,
		pin_memory=True if device.type != 'cpu' else False
	)
	print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

	# --- Model, Loss, Optimizer ---
	print("Initializing SkeletalGestureClassifier model...")
	hidden_dims_list = [int(d.strip()) for d in args.hidden_dims.split(',')] if args.hidden_dims else None
	model = SkeletalGestureClassifier(
		num_classes=num_classes,
		hidden_dims=hidden_dims_list,
		dropout_p=args.dropout_p
	).to(device)
	print(f"Model parameters: {model.count_parameters():,}")

	if args.wandb:
		wandb.watch(model, log="all", log_freq=100)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

	# --- Training Loop ---
	print(f"Starting training for {args.epochs} epochs...")
	for epoch in range(start_epoch, args.epochs):
		model.train()
		train_loss_epoch = 0.0
		processed_train_samples = 0
		train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
		
		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
		for batch in pbar:
			if batch is None: 
				continue

			landmarks, labels = batch
			landmarks = landmarks.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(landmarks)
			loss = criterion(outputs, labels)
			loss.backward()

			if args.clip_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
			
			optimizer.step()

			batch_size = landmarks.size(0)
			train_loss_epoch += loss.item() * batch_size
			processed_train_samples += batch_size
			train_acc_metric.update(outputs, labels)
			pbar.set_postfix(loss=loss.item())

		avg_train_loss = train_loss_epoch / processed_train_samples if processed_train_samples > 0 else 0
		train_accuracy = train_acc_metric.compute().item()
		train_acc_metric.reset()

		# --- Validation ---
		val_loss, val_accuracy, val_f1 = evaluate_skeletal_classifier(
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
				"learning_rate": optimizer.param_groups[0]['lr']
			})

		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			state = {
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict(),
				'val_f1': best_val_f1,
				'val_accuracy': val_accuracy,
				'num_classes': num_classes,
				'args': vars(args)
			}
			save_dir_project = wandb.config.project if args.wandb else args.project
			save_dir_run = wandb.config.run if args.wandb else args.run
			save_checkpoint(state, "best_val_f1.pth", save_dir_project, save_dir_run)
			print(f"Saved best validation model (F1: {best_val_f1:.4f}) at epoch {epoch+1}")

	# --- Test Set Evaluation ---
	print("Training complete. Evaluating on test set with best validation F1 model...")
	best_model_path = None
	try:
		save_dir_project = wandb.config.project if args.wandb else args.project
		save_dir_run = wandb.config.run if args.wandb else args.run
		best_model_path = os.path.join("checkpoints", save_dir_project, save_dir_run, "best_val_f1.pth")
		
		if os.path.exists(best_model_path):
			checkpoint = torch.load(best_model_path, map_location=device)
			ckpt_num_classes = checkpoint.get('num_classes', num_classes)
			if ckpt_num_classes != num_classes:
				print(f"Warning: Checkpoint num_classes ({ckpt_num_classes}) differs from current dataset ({num_classes}). Using current.")

			model.load_state_dict(checkpoint['model_state_dict'])
			print(f"Loaded best model from {best_model_path} (Epoch {checkpoint.get('epoch', 'N/A')}, Val F1: {checkpoint.get('val_f1', 0):.4f})")
		else:
			print(f"Warning: Best checkpoint {best_model_path} not found. Evaluating with model from last epoch.")
				
	except Exception as e:
		print(f"Error loading best checkpoint from {best_model_path if best_model_path else 'N/A'}: {e}. Evaluating with model from last epoch.")

	test_loss, test_accuracy, test_f1 = evaluate_skeletal_classifier(
		model, test_loader, criterion, device, num_classes, prefix="test"
	)

	print("--- Test Set Results ---")
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
	print("Skeletal classifier training finished.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train Skeletal Hand Gesture Classifier")

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	default_project_name = f"skeletal_hgr_{timestamp}"
	default_run_name = f"run_{timestamp}"

	parser.add_argument("--data_dir", type=str, default="data/hagrid_small", help="Root directory of the dataset")
	parser.add_argument("--hidden_dims", type=str, default="128,256,512,128", help="Comma-separated list of hidden layer dimensions for MLP")
	parser.add_argument("--dropout_p", type=float, default=0.3, help="Dropout probability in MLP")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
	parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
	parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum LR for CosineAnnealingLR")
	parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping (0 to disable)")
	parser.add_argument("--augment", action='store_true', help="Enable data augmentation (only affects image loading, landmarks are not directly augmented by this flag here but dataset might do it for images if it were used)")
	parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--wandb", action='store_true', help="Enable Weights & Biases logging")
	parser.add_argument("--project", type=str, default=default_project_name, help="Wandb project name")
	parser.add_argument("--run", type=str, default=default_run_name, help="Wandb run name")

	args = parser.parse_args()
	main(args)