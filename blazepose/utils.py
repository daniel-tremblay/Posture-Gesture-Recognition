import os
import json
import argparse
import numpy as np
import torch
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from types import SimpleNamespace
from typing import Tuple, Dict, Any, Optional
from model import default_blazepose_config

def visualize_sample(sample, keypoint_names, num_kpts):
	"""
	Visualize a sample from the COCO keypoints dataset.

	Args:
		sample (dict): Sample containing image, keypoints, and heatmap data
		keypoint_names (list): List of keypoint names for visualization
		num_kpts (int): Number of keypoints
	"""
	# Convert image back to numpy and denormalize
	image = sample['image'][0].numpy().transpose(1, 2, 0)
	keypoints = sample['keypoints'][0].numpy() # [17, 3] Target keypoints
	combined_hm_off = sample['combined_heatmap_offsets'][0].numpy() # [51, H/4, W/4]

	# Extract heatmaps from combined tensor
	heatmaps = combined_hm_off[0::3, :, :] # Select every 3rd channel starting from 0

	# Create a figure with two subplots
	fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Increased size for 3 plots
	ax1, ax2, ax3 = axes.flat

	# --- Plot image with keypoints ---
	ax1.imshow(image)
	ax1.set_title(f"Image with Target Keypoints (Vis: Scaled)\nImage ID: {sample['image_id'].item()}")
	ax1.axis('off')

	# Plot keypoints
	# Using target visibility for alpha/marker presence
	for i in range(num_kpts):
		visibility = keypoints[i, 2]
		if visibility > 0: # Plot if visible or occluded
			# Simple coloring for now
			color = plt.cm.viridis(i / num_kpts)
			# Scale size/alpha by visibility (0.5 for occluded, 1.0 for visible)
			size = 50 + 100 * visibility
			alpha = 0.5 + 0.5 * visibility
			ax1.scatter(
				keypoints[i, 0], keypoints[i, 1],
				c=[color], s=size, alpha=alpha,
				label=f"{keypoint_names[i]} (v={visibility:.1f})"
			)
	ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

	# --- Plot combined heatmaps ---
	ax2.set_title(f'Combined Heatmaps ({heatmaps.shape})')
	ax2.axis('off')
	if heatmaps.size > 0:
		combined_heatmap_display = np.max(heatmaps, axis=0)
		sns.heatmap(
			combined_heatmap_display, ax=ax2, cmap='rocket', cbar=True,
			square=True, xticklabels=False, yticklabels=False,
			cbar_kws={'label': 'Confidence'}
		)
	else:
		ax2.text(0.5, 0.5, 'No Heatmaps Generated', ha='center', va='center')

	# --- Plot Combined Offsets (Example: x-offset for keypoint 0) ---
	ax3.set_title(f'Offset dx Map (Nose, idx=0)')
	ax3.axis('off')
	# Extract dx offset for keypoint 0 (channel 1)
	if combined_hm_off.shape[0] > 1:
		offset_dx_nose = combined_hm_off[1, :, :]
		if offset_dx_nose.size > 0:
			# Use a diverging colormap for offsets
			sns.heatmap(
				offset_dx_nose, ax=ax3, cmap='coolwarm', cbar=True,
				square=True, xticklabels=False, yticklabels=False, center=0,
				cbar_kws={'label': 'dx offset'}
			)
		else:
			ax3.text(0.5, 0.5, 'No Offsets Generated', ha='center', va='center')
	else:
		ax3.text(0.5, 0.5, 'Offsets Not Available', ha='center', va='center')

	plt.tight_layout()
	plt.show() 

def load_train_config(
	config_path: str = 'config/train.json'
) -> Optional[Dict]:
	"""Loads training configuration from a JSON file.

	Args:
		config_path (str): Path to the configuration JSON file, relative to this script's directory.

	Returns:
		Dict | None: Configuration dictionary or None if loading fails.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	absolute_config_path = os.path.join(script_dir, config_path)
	try:
		with open(absolute_config_path, 'r') as f:
			config_dict = json.load(f)
		print(f"Loaded training configuration from {absolute_config_path}")
		return config_dict
	except FileNotFoundError:
		print(f"Warning: Configuration file not found at {absolute_config_path}. Returning None.")
		return None
	except json.JSONDecodeError:
		print(f"Error: Could not parse JSON configuration file at {absolute_config_path}")
		return None

def get_config(parser: argparse.ArgumentParser) -> Dict:
	"""
	Loads base configs, adds arguments to the provided parser using base configs
	as defaults, parses args, and constructs the final nested config object.

	Priority: command-line args > train.json config > default model config.

	Args:
		parser (argparse.ArgumentParser): An initialized argument parser instance.

	Returns:
		Dict: The final merged configuration dictionary.
	"""
	train_cfg = load_train_config() # Load from config/train.json (Dict or None)
	model_cfg_dict = default_blazepose_config() # Get defaults from model.py (dict)

	# Use a helper for cleaner default value retrieval
	def get_default_value(section_names, key, default_value):
		val = default_value
		current_level = train_cfg
		if current_level:
			try:
				for section in section_names:
					current_level = current_level[section]
				val = current_level[key]
			except (KeyError, TypeError): # Handle missing keys or non-dict levels
				pass # Value not found in train_cfg
		return val if val is not None else default_value


	# --- Add arguments to the parser ---
	# Default name includes timestamp
	default_run_name = f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

	# (Paths & Run Identification)
	parser.add_argument(
		'--coco_dir', type=str,
		default=get_default_value(['data'], 'coco_dir', "data/coco_keypoints"),
		help='Path to COCO dataset directory'
	)
	parser.add_argument(
		'--name', type=str,
		default=get_default_value(['logging'], 'name', default_run_name),
		help='Identifier for the run (used for checkpoints and wandb)'
	)

	# (Training Hyperparameters)
	parser.add_argument(
		'--lr', type=float,
		default=get_default_value(['optimizer'], 'learning_rate', 1e-3),
		help='Initial learning rate'
	)
	parser.add_argument(
		'--weight_decay', type=float,
		default=get_default_value(['optimizer'], 'weight_decay', 1e-4),
		help='Optimizer weight decay'
	)
	parser.add_argument(
		'--epochs', type=int,
		default=get_default_value(['training'], 'epochs', 10),
		help='Number of training epochs'
	)
	parser.add_argument(
		'--batch_size', type=int,
		default=get_default_value(['data'], 'batch_size', 64),
		help='Training batch size'
	)
	parser.add_argument(
		'--val_split', type=float,
		default=get_default_value(['data'], 'val_split', 0.1),
		help='Fraction of train set used for validation'
	)
	parser.add_argument(
		'--num_workers', type=int,
		default=get_default_value(['data'], 'num_workers', 2),
		help='Number of workers for DataLoader'
	)
	parser.add_argument(
		'--seed', type=int,
		default=get_default_value(['training'], 'seed', 42),
		help='Random seed for reproducibility'
	)

	# (Data Specific)
	parser.add_argument(
		'--max_samples', type=int,
		default=get_default_value(['data'], 'max_samples', None),
		help='Limit dataset size for quick testing (optional)'
	)
	parser.add_argument(
		'--grayscale_prob', type=float,
		default=get_default_value(['data'], 'grayscale_prob', 0.1),
		help='Probability of applying grayscale augmentation (0.0-1.0)'
	)
	parser.add_argument(
		'--visibility_handling', type=str,
		default=get_default_value(['data'], 'visibility_handling', 'scaled'),
		choices=['scaled', 'binary'],
		help='How to handle keypoint visibility (scaled or binary)'
	)
	parser.add_argument(
		'--occlusion_prob', type=float,
		default=get_default_value(['data'], 'occlusion_prob', 0.0),
		help='Probability of applying random occlusion (0.0-1.0)'
	)
	parser.add_argument(
		'--max_occlusions', type=int,
		default=get_default_value(['data'], 'max_occlusions', 2),
		help='Maximum number of occluding rectangles per image'
	)
	parser.add_argument(
		'--occlusion_size_range', type=float, nargs=2,
		default=get_default_value(['data'], 'occlusion_size_range', [0.05, 0.2]),
		help='Range of occlusion size as a fraction of image dimension (min max)'
	)
	parser.add_argument(
		'--augment', action=argparse.BooleanOptionalAction,
		default=get_default_value(['data'], 'augment', False),
		help='Enable/disable data augmentation'
	)

	# (Loss Weights)
	parser.add_argument(
		'--w_hm', type=float,
		default=get_default_value(['loss'], 'w_hm', 1.0),
		help='Weight for heatmap loss (float)'
	)
	parser.add_argument(
		'--w_off', type=float,
		default=get_default_value(['loss'], 'w_off', 1.0),
		help='Weight for offset loss (float)'
	)
	parser.add_argument(
		'--w_kp', type=float,
		default=get_default_value(['loss'], 'w_kp', 1.0),
		help='Weight for keypoint regression loss (float)'
	)
	parser.add_argument(
		'--offset_loss_type', type=str,
		default=get_default_value(['loss'], 'offset_loss_type', 'l1'),
		choices=['l1', 'l2', 'smooth_l1'],
		help='Loss type for offset regression'
	)
	parser.add_argument(
		'--kp_loss_type', type=str,
		default=get_default_value(['loss'], 'kp_loss_type', 'l1'),
		choices=['l1', 'l2', 'smooth_l1'],
		help='Loss type for keypoint regression'
	)

	# (Logging & Saving)
	parser.add_argument(
		'--log_interval', type=int,
		default=get_default_value(['logging'], 'log_interval', 50),
		help='Log training loss every N batches (int)'
	)
	parser.add_argument(
		'--wandb', action=argparse.BooleanOptionalAction,
		default=get_default_value(['logging'], 'wandb', False),
		help='Enable Weights & Biases logging '
	)

	# Parse Arguments
	args = parser.parse_args()

	# Construct Derived Values
	checkpoint_dir = f'papers/blaze-pose/checkpoints/{args.name}/{default_run_name}'
	wandb_project_name = args.name if args.wandb else None

	# --- Construct final config object ---
	config = {
		'data': {
			'coco_dir': args.coco_dir,
			'batch_size': args.batch_size,
			'num_workers': args.num_workers,
			'max_samples': args.max_samples,
			'val_split': args.val_split,
			'grayscale_prob': args.grayscale_prob,
			'visibility_handling': args.visibility_handling,
			'augment': args.augment,
			'occlusion_prob': args.occlusion_prob,
			'max_occlusions': args.max_occlusions,
			'occlusion_size_range': tuple(args.occlusion_size_range)
		},
		'training': {
			'epochs': args.epochs,
			'seed': args.seed,
		},
		'optimizer': {
			'learning_rate': args.lr,
			'weight_decay': args.weight_decay,
		},
		'loss': {
			'w_hm': args.w_hm,
			'w_off': args.w_off,
			'w_kp': args.w_kp,
			'offset_loss_type': args.offset_loss_type,
			'kp_loss_type': args.kp_loss_type,
		},
		'logging': {
			'name': args.name,
			'run_name': default_run_name,
			'wandb': args.wandb,
			'wandb_project': wandb_project_name,
			'wandb_run_name': default_run_name,
			'log_interval': args.log_interval,
			'checkpoint_dir': checkpoint_dir,
		},
		'model': model_cfg_dict
	}

	return config