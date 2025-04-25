import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader

from landmark import MediapipeHandsLandmarks
from dataset import HagridDataset, collate_filter_none
from torch_utils import get_device, optimize_for_device

def calculate_avg_distance(
	predicted_outputs: torch.Tensor, 
	true_outputs: torch.Tensor, 
	scale_factor: float = 256.0
) -> torch.Tensor:
	"""
	Calculates average distance in 256x256 pixel space.

	Args:
		predicted_outputs (torch.Tensor): Predictions, shape [B, N_landmarks, 2], normalized [0, 1].
		true_outputs (torch.Tensor): Ground truths, shape [B, N_landmarks, 2], normalized [0, 1].
		scale_factor (float): The factor to scale coordinates to pixel space (e.g., image width/height).

	Returns:
		torch.Tensor: Scalar tensor representing the batch average distance in pixels.
	"""
	if predicted_outputs.shape != true_outputs.shape:
		raise ValueError(f"Predictions shape {predicted_outputs.shape} must match GT shape {true_outputs.shape}")
	if len(predicted_outputs.shape) != 3 or predicted_outputs.shape[2] != 2:
		raise ValueError(f"Input tensors must have shape [B, N_landmarks, 2], got {predicted_outputs.shape}")
	if scale_factor <= 0:
		raise ValueError("scale_factor must be positive")

	predicted_outputs = predicted_outputs.float()
	true_outputs = true_outputs.float()

	# Convert coordinates from [0, 1] to [0-256]
	predicted_outputs_px = predicted_outputs * scale_factor
	true_outputs_px = true_outputs * scale_factor
	
	# Calculate Euclidean distance per landmark in pixel space
	distances_px = torch.norm(predicted_outputs_px - true_outputs_px, dim=2) # [B, N_landmarks]
	batch_avg_distance = torch.mean(distances_px)
	return batch_avg_distance # [B]
	
def calculate_avg_distance_per_sample(
	predicted_outputs: torch.Tensor,
	true_outputs: torch.Tensor,
	scale_factor: float = 256.0
) -> torch.Tensor:
	"""
	Calculates average distance per sample in 256x256 pixel space.

	Args:
		predicted_outputs (torch.Tensor): Predictions, shape [B, N_landmarks, 2], normalized [0, 1].
		true_outputs (torch.Tensor): Ground truths, shape [B, N_landmarks, 2], normalized [0, 1].
		scale_factor (float): The factor to scale coordinates to pixel space (e.g., image width/height).

	Returns:
		torch.Tensor: Tensor of shape [B] representing the average distance in pixels for each sample.
	"""
	if predicted_outputs.shape != true_outputs.shape:
		raise ValueError(f"Predictions shape {predicted_outputs.shape} must match GT shape {true_outputs.shape}")
	if len(predicted_outputs.shape) != 3 or predicted_outputs.shape[2] != 2:
		raise ValueError(f"Input tensors must have shape [B, N_landmarks, 2], got {predicted_outputs.shape}")
	if scale_factor <= 0:
		raise ValueError("scale_factor must be positive")

	predicted_outputs_px = predicted_outputs.float() * scale_factor
	true_outputs_px = true_outputs.float() * scale_factor

	# Calculate Euclidean distance per landmark in pixel space
	distances_px = torch.norm(predicted_outputs_px - true_outputs_px, dim=2) # [B, N_landmarks]
	sample_avg_distance = torch.mean(distances_px, dim=1) # [B]
	return sample_avg_distance

def calculate_pck(
	predicted_outputs: torch.Tensor,
	true_outputs: torch.Tensor,
	threshold: float = 20.0,
	scale_factor: float = 256.0
) -> float:
	"""
	Calculates the Percentage of Correct Keypoints (PCK) metric.

	Args:
		predicted_outputs (torch.Tensor): Predictions, shape [B, N_landmarks, 2], normalized [0, 1].
		true_outputs (torch.Tensor): Ground truths, shape [B, N_landmarks, 2], normalized [0, 1].
		threshold (float): The error tolerance threshold in pixels. Landmarks with error <= threshold are considered correct.
		scale_factor (float): The factor to scale coordinates to pixel space.

	Returns:
		float: The PCK value (proportion of correct keypoints).
	"""
	if predicted_outputs.shape != true_outputs.shape:
		raise ValueError("Predictions shape must match GT shape")
	if len(predicted_outputs.shape) != 3 or predicted_outputs.shape[2] != 2:
		raise ValueError("Input tensors must have shape [B, N_landmarks, 2]")
	if threshold < 0:
		raise ValueError("Threshold must be non-negative")
	if scale_factor <= 0:
		raise ValueError("scale_factor must be positive")

	predicted_outputs_px = predicted_outputs.float() * scale_factor
	true_outputs_px = true_outputs.float() * scale_factor

	# Calculate Euclidean distance per landmark in pixel space
	distances_px = torch.norm(predicted_outputs_px - true_outputs_px, dim=2) # Shape: [B, N_landmarks]
	correct_keypoints = distances_px <= threshold

	# Calculate PCK
	pck = torch.mean(correct_keypoints.float()).item()
	return pck

def evaluate_model(
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	device: torch.device,
	loss_fn: torch.nn.Module,
	scale_factor: float = 256.0,
	pck_threshold: float = 20.0
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
	"""
	Evaluates the model on a given dataset split.

	Args:
		model (torch.nn.Module): The model to evaluate.
		dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation set.
		device (torch.device): Device to run evaluation on.
		loss_fn (torch.nn.Module): The loss function used for training (e.g., MSELoss).
		scale_factor (float): Factor to scale coordinates to pixel space (default 256.0).
		pck_threshold (float): Threshold for PCK calculation in pixels (default 20.0).

	Returns:
		tuple[float, float, float, torch.Tensor, torch.Tensor]: Average loss, average pixel distance, and PCK score, and all predictions and targets.
	"""
	model.eval()
	total_loss = 0.0
	all_preds = []
	all_targets = []

	with torch.no_grad():
		for batch in dataloader:
			if batch is None:
				continue
			images, landmarks_gt = batch
			images = images.to(device)
			landmarks_gt = landmarks_gt.to(device)

			landmarks_pred = model(images)

			loss = loss_fn(landmarks_pred, landmarks_gt)
			total_loss += loss.item() * images.size(0) # Accumulate loss weighted by batch size
			all_preds.append(landmarks_pred.cpu())
			all_targets.append(landmarks_gt.cpu())

	if not all_preds:
		print("Warning: No valid batches found during evaluation")
		return 0.0, 0.0, 0.0, torch.tensor([]), torch.tensor([])

	all_preds = torch.cat(all_preds, dim=0)
	all_targets = torch.cat(all_targets, dim=0)

	# Calculate metrics
	avg_loss = total_loss / len(dataloader.dataset)
	avg_distance = calculate_avg_distance(all_preds, all_targets, scale_factor)
	pck = calculate_pck(all_preds, all_targets, threshold=pck_threshold, scale_factor=scale_factor)

	return avg_loss, avg_distance.item(), pck, all_preds, all_targets

def evaluate_model_with_error_tracking(
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	device: torch.device,
	loss_fn: torch.nn.Module,
	scale_factor: float = 256.0,
	pck_threshold: float = 20.0
) -> tuple[float, float, float, torch.Tensor, torch.Tensor, List[Tuple[int, float]]]:
	"""
	Evaluates the model, calculates overall metrics, and tracks per-sample errors.

	Args:
		model (torch.nn.Module): The model to evaluate.
		dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation set.
		device (torch.device): Device to run evaluation on.
		loss_fn (torch.nn.Module): The loss function (e.g., SmoothL1Loss).
		scale_factor (float): Factor to scale coordinates to pixel space.
		pck_threshold (float): Threshold for PCK calculation in pixels.

	Returns:
		tuple:
			- float: Average loss over the dataset.
			- float: Overall average pixel distance.
			- float: Overall PCK score.
			- torch.Tensor: All predicted landmarks.
			- torch.Tensor: All ground truth landmarks.
			- List[Tuple[int, float]]: List of tuples (sample_index, avg_distance_px) sorted by distance descending.
	"""
	model.eval()
	total_loss = 0.0
	all_preds = []
	all_targets = []
	sample_errors = [] # (original_index, error)
	current_index = 0

	with torch.no_grad():
		progress_bar = tqdm(dataloader, desc="Evaluating", total=len(dataloader))
		for batch_idx, batch in enumerate(progress_bar):
			if batch is None:
				print(f"Warning: Skipping None batch at index {batch_idx}")
				batch_size = dataloader.batch_size if dataloader.batch_size else 1
				current_index += batch_size
				continue

			images, landmarks_gt = batch
			batch_size = images.size(0)
			images = images.to(device)
			landmarks_gt = landmarks_gt.to(device)

			landmarks_pred = model(images)

			loss = loss_fn(landmarks_pred, landmarks_gt)
			total_loss += loss.item() * batch_size

			# Calculate per-sample average distance for this batch
			batch_sample_avg_dist = calculate_avg_distance_per_sample(
				landmarks_pred.cpu(), landmarks_gt.cpu(), scale_factor
			)

			for i in range(batch_size):
				original_dataset_index = current_index + i
				sample_errors.append((original_dataset_index, batch_sample_avg_dist[i].item()))

			all_preds.append(landmarks_pred.cpu())
			all_targets.append(landmarks_gt.cpu())
			current_index += batch_size

	if not all_preds:
		print("Warning: No valid batches found during evaluation")
		return 0.0, 0.0, 0.0, torch.tensor([]), torch.tensor([]), []

	all_preds = torch.cat(all_preds, dim=0)
	all_targets = torch.cat(all_targets, dim=0)

	num_evaluated_samples = all_preds.shape[0]
	if num_evaluated_samples == 0:
		return 0.0, 0.0, 0.0, all_preds, all_targets, []

	avg_loss = total_loss / num_evaluated_samples
	overall_avg_distance = calculate_avg_distance(all_preds, all_targets, scale_factor)
	overall_pck = calculate_pck(all_preds, all_targets, threshold=pck_threshold, scale_factor=scale_factor)
	sample_errors.sort(key=lambda item: item[1], reverse=True)

	return avg_loss, overall_avg_distance.item(), overall_pck, all_preds, all_targets, sample_errors

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate MediaPipe Hands Landmark Model")
	parser.add_argument(
		"--weights",
		type=str,
		required=True,
		help="Path to the trained model checkpoint (.pth)"
	)
	parser.add_argument(
		"--data_dir",
		type=str,
		default="data/hagrid_small",
		help="Root directory of the Hagrid dataset"
	)
	parser.add_argument(
		"--split",
		type=str,
		default="test",
		help="Dataset split to evaluate on (default: test)"
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=32,
		help="Batch size for evaluation"
	)
	parser.add_argument(
		"--scale_factor",
		type=float,
		default=256.0,
		help="Scale factor for pixel space conversion (image size)"
	)
	parser.add_argument(
		"--pck_threshold",
		type=float,
		default=10.0,
		help="Threshold for PCK calculation in pixels"
	)
	parser.add_argument(
		"--num_workers",
		type=int,
		default=1,
		help="Number of workers for DataLoader"
	)
	parser.add_argument(
		"--top_n",
		type=int,
		default=10,
		help="Number of samples with highest errors to display"
	)
	args = parser.parse_args()

	device = get_device()
	optimize_for_device(device)
	print(f"Using device: {device}")

	print(f"Loading model from: {args.weights}")
	model = MediapipeHandsLandmarks()
	try:
		checkpoint = torch.load(args.weights, map_location=device)
		if 'model_state_dict' in checkpoint:
			model.load_state_dict(checkpoint['model_state_dict'])
		else:
			model.load_state_dict(checkpoint)
		print("Model loaded successfully.")
	except FileNotFoundError:
		print(f"Error: Model checkpoint not found at {args.model_path}")
		exit(1)
	except Exception as e:
		print(f"Error loading model checkpoint: {e}")
		exit(1)

	model.to(device)
	model.eval()

	print(f"Loading dataset from: {args.data_dir}, split: {args.split}")
	try:
		test_dataset = HagridDataset(
			root=args.data_dir,
			split=args.split,
			augment=False
		)
		if len(test_dataset) == 0:
			print(f"Error: No samples found in the '{args.split}' split at {args.data_dir}. Check dataset structure.")
			exit(1)

		test_loader = DataLoader(
			test_dataset,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
			collate_fn=collate_filter_none,
			pin_memory=True if device.type == 'cuda' else False
		)
		print(f"Loaded {len(test_dataset)} samples for evaluation.")
	except FileNotFoundError:
		print(f"Error: Dataset directory or annotation file not found at {args.data_dir}")
		exit(1)
	except Exception as e:
		print(f"Error loading dataset: {e}")
		exit(1)

	loss_fn = nn.SmoothL1Loss()

	print("\nStarting evaluation...")
	avg_loss, avg_distance, pck, all_preds, all_targets, sorted_errors = evaluate_model_with_error_tracking(
		model=model,
		dataloader=test_loader,
		device=device,
		loss_fn=loss_fn,
		pck_threshold=args.pck_threshold
	)

	print("\n--- Overall Metrics ---")
	print(f"  Average Loss: {avg_loss:.4f}")
	print(f"  Avg Distance (pixels): {avg_distance:.2f}")
	print(f"  PCK@{args.pck_threshold}px: {pck * 100:.2f}%")

	# Calculate Per-Landmark Metrics
	if all_preds.numel() > 0 and all_targets.numel() > 0:
		print("\n--- Per-Landmark Metrics ---")
		num_landmarks = all_preds.shape[1]

		all_preds_px = all_preds.float() * args.scale_factor
		all_targets_px = all_targets.float() * args.scale_factor

		distances_px = torch.norm(all_preds_px - all_targets_px, dim=2) # Shape: [N_total, N_landmarks]
		avg_dist_per_landmark = torch.mean(distances_px, dim=0) # Shape: [N_landmarks]

		correct_keypoints = distances_px <= args.pck_threshold # Shape: [N_total, N_landmarks]
		pck_per_landmark = torch.mean(correct_keypoints.float(), dim=0) # Shape: [N_landmarks]

		print(f"  Metric calculated using PCK threshold = {args.pck_threshold} pixels")
		print("  Landmark | Avg Distance (px) | PCK (%)")
		print("  ---------|-------------------|--------")
		for i in range(num_landmarks):
			print(f"  {i:<8} | {avg_dist_per_landmark[i].item():<17.2f} | {pck_per_landmark[i].item() * 100:<7.2f}")
	else:
		print("\nSkipping per-landmark metrics as no predictions were generated.")

	# Display samples with the highest errors
	if sorted_errors:
		print(f"\n--- Top {min(args.top_n, len(sorted_errors))} Samples with Highest Avg Distance Error ---")
		print("  Dataset Index | Avg Distance (px)")
		print("  -------------|-------------------")
		for i in range(min(args.top_n, len(sorted_errors))):
			idx, error = sorted_errors[i]
			print(f"  {idx:<13} | {error:<18.2f}")
	else:
		print("\nNo samples evaluated to show error ranking.")

	print("\nEvaluation finished.")