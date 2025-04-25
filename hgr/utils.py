import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Union, Dict

# --DATASET UTILS--

def plot_hand_data(
	original_image: np.ndarray,
	cropped_image: np.ndarray,
	landmarks_crop: np.ndarray,
	bbox_original: Union[List[float], Tuple[float, float, float, float], np.ndarray],
	bbox_crop: Union[List[float], Tuple[float, float, float, float], np.ndarray],
	titles: Dict[str, str],
	landmark_color: str = 'cyan',
	landmark_size: int = 80,
	bbox_original_color: str = 'lime',
	bbox_crop_color: str = 'lime',
	show_labels: bool = True,
	figsize: Tuple[int, int] = (10, 5)
) -> None:
	"""
	Plots the original image with its bounding box and the cropped image with its bounding box and landmarks side-by-side.

	Args:
		original_image: The original image (H, W, 3) as a NumPy array (RGB, uint8).
		cropped_image: The cropped image (Hc, Wc, 3) as a NumPy array (RGB, uint8).
		landmarks_crop: 
			Hand landmarks with absolute coordinates relative to the cropped image (N, 2) as a NumPy array.
		bbox_original: 
			Bounding box [x1, y1, x2, y2] with absolute coordinates relative to the original image.
		bbox_crop: 
			Bounding box [x1, y1, x2, y2] with absolute coordinates relative to the cropped image.
		titles: A dictionary containing titles. Expected keys: 'title', 'original', 'cropped'.
		landmark_color: Color for the plotted landmarks.
		landmark_size: Size (area) for the plotted landmarks (scatter plot 's' parameter).
		bbox_original_color: Color for the bounding box on the original image.
		bbox_crop_color: Color for the bounding box on the cropped image.
		show_labels: Whether to display the index number next to each landmark on the cropped image.
		figsize: Figure size for the plot.
	"""
	fig, ax = plt.subplots(1, 2, figsize=figsize)
	fig.suptitle(titles['title'])
	
	# --- Original Image Plot ---
	ax[0].imshow(original_image)
	ax[0].set_title(titles['original'])
	ax[0].axis('off')
	
	# Draw bbox on original image
	x1_orig, y1_orig, x2_orig, y2_orig = bbox_original
	rect_orig = patches.Rectangle(
		(x1_orig, y1_orig), x2_orig - x1_orig, y2_orig - y1_orig, 
		linewidth=1, edgecolor=bbox_original_color, facecolor='none'
	)
	ax[0].add_patch(rect_orig)
	
	# --- Cropped Image Plot ---
	ax[1].imshow(cropped_image)
	ax[1].set_title(titles['cropped'])
	ax[1].axis('off')
	
	# Draw bbox on cropped image
	x1_crop, y1_crop, x2_crop, y2_crop = bbox_crop
	rect_crop = patches.Rectangle(
		(x1_crop, y1_crop), x2_crop - x1_crop, y2_crop - y1_crop,
		linewidth=2, edgecolor=bbox_crop_color, facecolor='none'
	)
	ax[1].add_patch(rect_crop)

	# Draw landmarks on cropped image
	for i, (x, y) in enumerate(landmarks_crop):
		ax[1].scatter(x, y, s=landmark_size, c=landmark_color, marker='.')
		# Add index next to the landmark
		if show_labels:
			ax[1].text(x + 2, y + 2, str(i), color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, pad=0.1))

	# Set plot limits to crop dimensions
	h_crop, w_crop, _ = cropped_image.shape 
	ax[1].set_xlim(0, w_crop)
	ax[1].set_ylim(h_crop, 0) # Flipped y-axis for image coordinates

	plt.tight_layout()
	plt.show()

def transform_bbox_to_crop(
	bbox_abs: Union[List[float], Tuple[float, float, float, float]],
	crop_coords: Tuple[float, float, float, float],
	scales: Tuple[float, float],
	crop_size: Union[int, Tuple[int, int]]
) -> List[float]:
	"""
	Transforms absolute bounding box coordinates from the original image to the resized crop coordinates.

	Args:
		bbox_abs: Bounding box [x1, y1, x2, y2] in absolute pixel coordinates of the original image.
		crop_coords: Absolute coordinates (left, top, right, bottom) of the crop in the original image.
		scales: Scaling factors (scale_x, scale_y) applied when resizing the crop.
		crop_size: Dimensions (height, width) of the resized crop, or a single int if square.

	Returns:
		List[float]: Bounding box [x1, y1, x2, y2] in absolute pixel coordinates of the resized crop.
	"""
	x1_abs, y1_abs, x2_abs, y2_abs = bbox_abs
	offset_x, offset_y, _, _ = crop_coords
	scale_x, scale_y = scales

	h_crop, w_crop = crop_size

	# Transform coordinates
	x1_crop_resized = (x1_abs - offset_x) * scale_x
	y1_crop_resized = (y1_abs - offset_y) * scale_y
	x2_crop_resized = (x2_abs - offset_x) * scale_x
	y2_crop_resized = (y2_abs - offset_y) * scale_y

	# Clamp coordinates to crop boundaries
	x1_crop_resized = max(0, min(w_crop - 1, x1_crop_resized))
	y1_crop_resized = max(0, min(h_crop - 1, y1_crop_resized))
	x2_crop_resized = max(x1_crop_resized, min(w_crop, x2_crop_resized))
	y2_crop_resized = max(y1_crop_resized, min(h_crop, y2_crop_resized))

	return [x1_crop_resized, y1_crop_resized, x2_crop_resized, y2_crop_resized]

# -- TRAINING UTILS --

def set_seed(seed):
	"""Sets the seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	print(f"Random seed set to {seed}")

def save_checkpoint(
	state: Dict,
	filename: str,
	project_name: str,
	run_name: str
) -> None:
	"""
	Saves the training checkpoints.

	Args:
		state (Dict): Dictionary containing model state, optimizer state, epoch, etc.
		filename (str): Name for the checkpoint file (e.g., 'best_train.pth', 'best_val.pth').
		project_name (str): Name of the Wandb project.
		run_name (str): Name of the Wandb run.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	checkpoint_dir = os.path.join(script_dir, "checkpoints", project_name, run_name)
	os.makedirs(checkpoint_dir, exist_ok=True)
	filepath = os.path.join(checkpoint_dir, filename)
	torch.save(state, filepath)