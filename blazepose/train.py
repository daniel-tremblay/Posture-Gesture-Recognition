import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import BlazePose
from dataset import COCOKeypointsDataset
from torch_utils import get_device
from utils import get_config

# --- Loss Function ---
class BlazePoseLoss(nn.Module):
	"""
	Combined loss function for BlazePose training.
	Includes Heatmap Loss, Offset Loss, and Keypoint Regression Loss.
	"""
	def __init__(
		self, 
		w_hm=1.0, 
		w_off=1.0, 
		w_kp=1.0, 
		offset_loss_type='l1', 
		kp_loss_type='l1'
	):
		"""
		Args:
			w_hm (float): Weight for heatmap loss.
			w_off (float): Weight for offset loss.
			w_kp (float): Weight for keypoint regression loss.
			offset_loss_type (str): 'l1' or 'smooth_l1' for offset loss.
			kp_loss_type (str): 'l1' or 'smooth_l1' for keypoint loss.
		"""
		super().__init__()
		self.w_hm = w_hm
		self.w_off = w_off
		self.w_kp = w_kp

		# Heatmap Loss (Mean Squared Error)
		self.heatmap_loss_fn = nn.MSELoss(reduction='mean')

		# Offset Loss (L1 or Smooth L1)
		if offset_loss_type == 'l1':
			self.offset_loss_fn = nn.L1Loss(reduction='none') # Calculate per element loss first
		elif offset_loss_type == 'smooth_l1':
			self.offset_loss_fn = nn.SmoothL1Loss(reduction='none', beta=1.0)
		else:
			raise ValueError(f"Unsupported offset_loss_type: {offset_loss_type}")

		# Keypoint Regression Loss (L1 or Smooth L1)
		if kp_loss_type == 'l1':
			self.keypoint_loss_fn = nn.L1Loss(reduction='none')
		elif kp_loss_type == 'smooth_l1':
			self.keypoint_loss_fn = nn.SmoothL1Loss(reduction='none', beta=1.0)
		else:
			 raise ValueError(f"Unsupported kp_loss_type: {kp_loss_type}")

	def forward(
		self, 
		pred_combined_hm_off, 
		pred_keypoints,
		target_combined_hm_off, 
		target_keypoints, 
		target_offset_weights
	):
		"""
		Calculate the combined loss.

		Args:
			pred_combined_hm_off (Tensor): Model output [B, K*3, H, W]
			pred_keypoints (Tensor): Model output [B, K, 3] (x, y, vis)
			target_combined_hm_off (Tensor): Ground truth [B, K*3, H, W]
			target_keypoints (Tensor): Ground truth [B, K, 3] (x, y, vis)
			target_offset_weights (Tensor): Weights for offset loss [B, K*2, H, W]

		Returns:
			Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss, dictionary of component losses.
		"""
		batch_size = pred_keypoints.shape[0]

		# --- 1. Split Predicted and Target Heatmaps/Offsets ---
		# Shape: [B, K, H, W]
		pred_heatmaps = pred_combined_hm_off[:, 0::3, :, :]
		target_heatmaps = target_combined_hm_off[:, 0::3, :, :]

		# Shape: [B, K*2, H, W] (dx0, dy0, dx1, dy1, ...)
		pred_offsets = torch.cat(
			(pred_combined_hm_off[:, 1::3, :, :], pred_combined_hm_off[:, 2::3, :, :]),
			dim=1
		)
		target_offsets = torch.cat(
			(target_combined_hm_off[:, 1::3, :, :], target_combined_hm_off[:, 2::3, :, :]),
			dim=1
		)
		# Ensure target_offset_weights matches the shape
		assert target_offset_weights.shape == target_offsets.shape, \
			   f"Offset weights shape {target_offset_weights.shape} mismatch offsets shape {target_offsets.shape}"

		# --- 2. Calculate Heatmap Loss ---
		# Compare predicted heatmaps directly to target Gaussian heatmaps
		heatmap_loss = self.heatmap_loss_fn(pred_heatmaps, target_heatmaps)

		# --- 3. Calculate Offset Loss ---
		# Calculate element-wise loss
		offset_loss_elementwise = self.offset_loss_fn(pred_offsets, target_offsets)
		# Apply weights (only calculate loss where GT offset exists)
		weighted_offset_loss = offset_loss_elementwise * target_offset_weights
		# Normalize by the number of valid offset pixels (sum of weights)
		# Add epsilon to prevent division by zero if no offsets are valid in a batch
		num_valid_offsets = torch.sum(target_offset_weights) + 1e-6
		offset_loss = torch.sum(weighted_offset_loss) / num_valid_offsets

		# --- 4. Calculate Keypoint Regression Loss ---
		# Only calculate loss for keypoints that are visible or occluded (v > 0 in original COCO)
		# Assuming 'scaled' visibility: v=0.5 (occluded), v=1.0 (visible)
		# Use target_keypoints[:, :, 2] > 0 as the mask
		keypoint_mask = (target_keypoints[:, :, 2:3] > 0).float() # Shape [B, K, 1], broadcastable

		# Compare predicted x,y coords with target x,y coords
		kp_loss_elementwise = self.keypoint_loss_fn(
			pred_keypoints[:, :, 0:2], # Compare only x, y
			target_keypoints[:, :, 0:2]
		)

		# Apply mask (loss for x and y depends on visibility)
		# Shape [B, K, 2] * [B, K, 1] -> [B, K, 2]
		masked_kp_loss = kp_loss_elementwise * keypoint_mask
		# Normalize by number of valid keypoints * 2 (for x and y)
		num_valid_keypoints = torch.sum(keypoint_mask) + 1e-6
		keypoint_loss = torch.sum(masked_kp_loss) / (num_valid_keypoints * 2) # Multiply by 2 as loss is for x and y

		# --- 5. Combine Losses ---
		total_loss = (
			self.w_hm * heatmap_loss +
			self.w_off * offset_loss +
			self.w_kp * keypoint_loss
		)

		component_losses = {
			'total_loss': total_loss.detach(),
			'heatmap_loss': heatmap_loss.detach(),
			'offset_loss': offset_loss.detach(),
			'keypoint_loss': keypoint_loss.detach()
		}

		return total_loss, component_losses

# --- Utility Functions ---
def save_checkpoint(
	state, 
	is_best, 
	filename='checkpoint.pt',
	best_filename='model_best.pt'
):
	"""Saves model checkpoint."""
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	torch.save(state, filename)
	if is_best:
		torch.save(state, best_filename)
		print(f" => Saved new best model to {best_filename}")

def set_seed(seed):
	"""Sets random seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		# Ensure deterministic behavior for CuDNN
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	# Check for MPS availability and set seed if possible/needed
	elif torch.backends.mps.is_available():
		 print("Setting seed for MPS (using torch.manual_seed)")
		 try:
		    torch.mps.manual_seed(seed)
		 except AttributeError:
		  	print("torch.mps.manual_seed not available, relying on torch.manual_seed.")

def collate_fn_skip_none(batch):
	"""
	Custom collate function that filters out None items from a batch
	before passing to default_collate.
	"""
	batch = list(filter(lambda x: x is not None, batch))
	if len(batch) == 0:
		return None # Return None if the whole batch failed
	# Use default collate to handle the rest
	return torch.utils.data.dataloader.default_collate(batch)

def evaluate_test_set(model, config, device):
	"""Evaluate the model on the COCO val2017 test set."""
	print("\nEvaluating on test set (val2017)...")
	
	# Create test dataset
	test_dataset = COCOKeypointsDataset(
		coco_dir=config['data']['coco_dir'],
		split='val2017',
		input_size=config['model']['input_size'],
		augment=False,
		visibility_handling=config['data']['visibility_handling']
	)
	
	# Create test dataloader
	test_loader = DataLoader(
		test_dataset,
		batch_size=config['data']['batch_size'],
		shuffle=False,
		num_workers=config['data']['num_workers'],
		pin_memory=True,
		collate_fn=collate_fn_skip_none
	)
	
	# Initialize metrics
	test_loss_accum = 0.0
	test_component_losses_accum = {}
	criterion = BlazePoseLoss(
		w_hm=config['loss']['w_hm'],
		w_off=config['loss']['w_off'],
		w_kp=config['loss']['w_kp'],
		offset_loss_type=config['loss']['offset_loss_type'],
		kp_loss_type=config['loss']['kp_loss_type']
	).to(device)
	
	# Evaluation loop
	model.eval()
	with torch.no_grad():
		pbar = tqdm(test_loader, desc="Testing")
		for batch in pbar:
			if batch is None:
				continue
				
			images = batch['image'].to(device, non_blocking=True)
			target_kpts = batch['keypoints'].to(device, non_blocking=True)
			target_hm_off = batch['combined_heatmap_offsets'].to(device, non_blocking=True)
			target_off_weights = batch['offset_weights'].to(device, non_blocking=True)
			
			# Forward pass
			pred_hm_off, pred_kpts = model(images, training=True)
			
			# Calculate loss
			loss, component_losses = criterion(
				pred_hm_off, 
				pred_kpts,
				target_hm_off, 
				target_kpts, 
				target_off_weights
			)
			
			# Accumulate losses
			test_loss_accum += loss.item()
			for key, value in component_losses.items():
				test_component_losses_accum[key] = test_component_losses_accum.get(key, 0.0) + value.item()
	
	# Calculate average losses
	avg_test_loss = test_loss_accum / len(test_loader) if len(test_loader) > 0 else 0.0
	avg_test_component_losses = {k: v / len(test_loader) for k, v in test_component_losses_accum.items()} if len(test_loader) > 0 else {}
	
	print("\nTest Set Results:")
	print(f"  Test Loss: {avg_test_loss:.4f}")
	for key, value in avg_test_component_losses.items():
		print(f"  {key}: {value:.4f}")
	
	# Log to wandb if enabled
	if config['logging']['wandb_project']:
		log_dict = {
			'test_loss': avg_test_loss,
		}
		for key, value in avg_test_component_losses.items():
			log_dict[f'test_{key}'] = value
		wandb.log(log_dict)
	
	return avg_test_loss, avg_test_component_losses

# --- Main Training Function ---
def main(config):
	"""Main training and validation function."""
	# --- Setup ---
	set_seed(config['training']['seed'])
	device = get_device()
	print(f"Using device: {device}")
	
	# Initialize WandB if enabled
	if config['logging']['wandb_project']:
		wandb.init(
			project=config['logging']['wandb_project'],
			config=config,
			name=config['logging']['run_name']
		)

	# --- Datasets and DataLoaders ---
	print("Loading datasets...")

	# Create the full training dataset instance
	full_train_dataset = COCOKeypointsDataset(
		coco_dir=config['data']['coco_dir'],
		split='train2017',
		input_size=config['model']['input_size'],
		augment=config['data']['augment'],
		max_samples=config['data']['max_samples'],
		visibility_handling=config['data']['visibility_handling'],
		grayscale_prob=config['data']['grayscale_prob']
	)

	# Split training dataset into actual training and validation sets
	val_size = int(config['data']['val_split'] * len(full_train_dataset))
	train_size = len(full_train_dataset) - val_size
	print(f"Splitting train2017: {train_size} training samples, {val_size} validation samples")
	try:
		train_dataset, val_dataset = random_split(
			full_train_dataset,
			[train_size, val_size],
			generator=torch.Generator().manual_seed(config['training']['seed'])
		)
	except ValueError as e:
		print(f"Error during dataset split: {e}")
		print("Ensure dataset is not empty and split sizes are valid.")
		return

	# Create DataLoaders
	train_loader = DataLoader(
		train_dataset,
		batch_size=config['data']['batch_size'],
		shuffle=True,
		num_workers=config['data']['num_workers'],
		pin_memory=True,
		collate_fn=collate_fn_skip_none,
		drop_last=True
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=config['data']['batch_size'],
		shuffle=False,
		num_workers=config['data']['num_workers'],
		pin_memory=True,
		collate_fn=collate_fn_skip_none
	)
	print("DataLoaders created.")

	# --- Model ---
	print("Initializing model...")
	model = BlazePose(config=config['model']).to(device)
	if config['logging']['wandb_project']:
		wandb.watch(model, log='all', log_freq=100)

	# --- Loss, Optimizer, Scheduler ---
	criterion = BlazePoseLoss(
		w_hm=config['loss']['w_hm'],
		w_off=config['loss']['w_off'],
		w_kp=config['loss']['w_kp']
	).to(device)

	optimizer = optim.AdamW(
		model.parameters(),
		lr=config['optimizer']['learning_rate'],
		weight_decay=config['optimizer']['weight_decay']
	)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode='min',
		factor=0.1,
		patience=5,
		verbose=True
	)

	# --- Checkpoint Loading ---
	best_val_loss = float('inf')
	start_epoch = 0

	print(f"Starting training for {config['training']['epochs']} epochs...")
	for epoch in range(start_epoch, config['training']['epochs']):
		start_time = time.time()

		# --- Training Phase ---
		model.train()
		train_loss_accum = 0.0
		train_component_losses_accum = {}

		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
		for i, batch in enumerate(pbar):
			 # Skip if collate_fn returned None due to all samples failing
			if batch is None:
				print(f"Warning: Skipping batch {i} due to all samples failing to load.")
				continue

			images = batch['image'].to(device, non_blocking=True)
			target_kpts = batch['keypoints'].to(device, non_blocking=True)
			target_hm_off = batch['combined_heatmap_offsets'].to(device, non_blocking=True)
			target_off_weights = batch['offset_weights'].to(device, non_blocking=True)

			# Zero gradients
			optimizer.zero_grad()

			# Forward pass
			pred_hm_off, pred_kpts = model(images, training=True)

			# Calculate loss
			loss, component_losses = criterion(
				pred_hm_off, 
				pred_kpts,
				target_hm_off, 
				target_kpts, 
				target_off_weights
			)

			# Backward pass
			loss.backward()

			# Optimizer step
			optimizer.step()

			# Accumulate losses
			train_loss_accum += loss.item()
			for key, value in component_losses.items():
				train_component_losses_accum[key] = train_component_losses_accum.get(key, 0.0) + value.item()

			# Update progress bar
			if (i + 1) % config['logging']['log_interval'] == 0:
				current_lr = optimizer.param_groups[0]['lr']
				avg_loss = train_loss_accum / (i + 1)
				pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.1e}")
				if config['logging']['wandb_project']:
					wandb.log({
						'train_batch_loss': loss.item(),
						'learning_rate': current_lr,
						'epoch': epoch + (i+1)/len(train_loader), # Log fractional epoch
						'batch_idx': i + epoch * len(train_loader)
					})

		avg_train_loss = train_loss_accum / len(train_loader) if len(train_loader) > 0 else 0.0
		avg_train_component_losses = {k: v / len(train_loader) for k, v in train_component_losses_accum.items()} if len(train_loader) > 0 else {}

		# --- Validation Phase ---
		model.eval()
		val_loss_accum = 0.0
		val_component_losses_accum = {}

		with torch.no_grad():
			pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
			for batch in pbar_val:
				if batch is None: continue # Skip failed batches

				images = batch['image'].to(device, non_blocking=True)
				target_kpts = batch['keypoints'].to(device, non_blocking=True)
				target_hm_off = batch['combined_heatmap_offsets'].to(device, non_blocking=True)
				target_off_weights = batch['offset_weights'].to(device, non_blocking=True)

				# Forward pass - still need both outputs for loss calculation
				pred_hm_off, pred_kpts = model(images, training=True)

				# Calculate loss
				loss, component_losses = criterion(
					pred_hm_off, 
					pred_kpts,
					target_hm_off, 
					target_kpts, 
					target_off_weights
				)

				val_loss_accum += loss.item()
				for key, value in component_losses.items():
					val_component_losses_accum[key] = val_component_losses_accum.get(key, 0.0) + value.item()

		avg_val_loss = val_loss_accum / len(val_loader) if len(val_loader) > 0 else 0.0
		avg_val_component_losses = {k: v / len(val_loader) for k, v in val_component_losses_accum.items()} if len(val_loader) > 0 else {}


		# --- Epoch End ---
		epoch_time = time.time() - start_time
		print(f"Epoch {epoch+1}/{config['training']['epochs']} Summary:")
		print(f"  Train Loss: {avg_train_loss:.4f}")
		print(f"  Val Loss:   {avg_val_loss:.4f}")
		print(f"  Time:       {epoch_time:.2f}s")

		# Log metrics
		if config['logging']['wandb_project']:
			log_dict = {
				'epoch': epoch + 1,
				'train_loss_epoch': avg_train_loss,
				'val_loss_epoch': avg_val_loss,
				'epoch_time_s': epoch_time,
			}
			# Add component losses
			for key, value in avg_train_component_losses.items():
				log_dict[f'train_{key}'] = value
			for key, value in avg_val_component_losses.items():
				log_dict[f'val_{key}'] = value
			wandb.log(log_dict)

		# Scheduler step (based on validation loss)
		scheduler.step(avg_val_loss)

		# Save checkpoint
		is_best = avg_val_loss < best_val_loss
		best_val_loss = min(avg_val_loss, best_val_loss)

		checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], f"epoch_{epoch+1}_checkpoint.pt")
		best_model_path = os.path.join(config['logging']['checkpoint_dir'], "model_best.pt")

		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'scheduler': scheduler.state_dict(),
			'best_val_loss': best_val_loss,
			'config': config,
		}, is_best, filename=checkpoint_path, best_filename=best_model_path)

	print("Training finished.")
	
	# Evaluate on test set
	evaluate_test_set(model, config, device)
	
	if config['logging']['wandb_project']:
		wandb.finish()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='BlazePose Training Script')
	config = get_config(parser)

	# Check required arguments
	if config['data']['coco_dir'] is None:
		raise ValueError("--coco_dir is required either as a command-line argument or in the train.json file.")

	# Create checkpoint directory
	os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)

	# Start training Bla
	main(config)