import os
import cv2
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from typing import Tuple, Dict, List, Any
from torch.utils.data import Dataset, DataLoader
from utils import visualize_sample

class COCOKeypointsDataset(Dataset):
	"""
	COCO keypoints dataset loader
	Generates images, keypoints, heatmaps, offsets, and offset weights.
	"""
	def __init__(
		self, 
		coco_dir, 
		split='train2017',
		input_size=256,
		heatmap_size_ratio: int = 4, # Heatmap size = input_size / ratio
		sigma: float = 2.0,          # Gaussian sigma for heatmaps
		offset_radius: int = 3,      # Radius for offset map generation
		max_samples=None, 
		augment=False,
		visibility_handling: str = 'scaled', # 'binary' or 'scaled'
		grayscale_prob: float = 0.3, # Probability of applying grayscale augmentation
		occlusion_prob: float = 0.2,   # Probability of applying occlusion
		max_occlusions: int = 2,     # Max number of occluding rectangles
		occlusion_size_range: Tuple[float, float] = (0.1, 0.3) # Size range as fraction of image dim
	):
		if not os.path.exists(coco_dir):
			raise ValueError(f"COCO directory does not exist: {coco_dir}")
		if split not in ['train2017', 'val2017']:
			raise ValueError(f"Invalid split: {split}")
		if input_size <= 0:
			raise ValueError(f"Invalid input_size: {input_size}")
		if not 0 <= grayscale_prob <= 1:
			raise ValueError(f"grayscale_prob must be between 0 and 1, got {grayscale_prob}")
		if not 0 <= occlusion_prob <= 1:
			raise ValueError(f"occlusion_prob must be between 0 and 1, got {occlusion_prob}")
		if max_occlusions < 0:
			raise ValueError(f"max_occlusions must be non-negative, got {max_occlusions}")
		if not (0 < occlusion_size_range[0] <= occlusion_size_range[1] <= 1):
			raise ValueError(f"Invalid occlusion_size_range: {occlusion_size_range}")

		self.coco_dir = coco_dir
		self.split = split
		self.input_size = input_size
		self.heatmap_size = input_size // heatmap_size_ratio
		self.sigma = sigma
		self.offset_radius = offset_radius
		self.augment = augment
		self.visibility_handling = visibility_handling
		self.grayscale_prob = grayscale_prob
		self.num_keypoints = 17 # COCO Keypoints default
		self.occlusion_prob = occlusion_prob
		self.max_occlusions = max_occlusions
		self.occlusion_size_range = occlusion_size_range
		
		# --- Albumentations Augmentation Pipeline ---
		self.transform = None
		if self.augment:
			# Define probabilities and limits for geometric augmentations
			shift_limit = 0.0625 # Max percentage shift
			scale_limit = 0.15   # Max percentage scale change (+/-)
			rotate_limit = 40    # Max rotation degrees (+/-)
			ssr_prob = 0.75      # Probability of applying ShiftScaleRotate
			flip_prob = 0.5      # Probability of horizontal flip
			brightness_contrast_prob = 0.5
			# Occlusion probability is handled by self.occlusion_prob

			self.transform = A.Compose([
				A.HorizontalFlip(p=flip_prob),
				A.Affine(
					scale=(1.0 - scale_limit, 1.0 + scale_limit),
					translate_percent=(-shift_limit, shift_limit),
					rotate=(-rotate_limit, rotate_limit),
					p=ssr_prob,
					border_mode=cv2.BORDER_CONSTANT, # Use border_mode
					cval=0 # Use cval for constant fill value
				),
				A.RandomBrightnessContrast(p=brightness_contrast_prob),
				A.ToGray(p=self.grayscale_prob),
			], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

		# Load COCO annotations
		ann_file = os.path.join(coco_dir, 'annotations', f'person_keypoints_{split}.json')
		self.coco = COCO(ann_file)
		
		# Get person category id
		cat_ids = self.coco.getCatIds(catNms=['person'])

		# Get all image ids with people
		img_ids = self.coco.getImgIds(catIds=cat_ids)

		# Collect samples with at least one labeled keypoint (v=1 or v=2)
		self.samples: List[Dict[str, Any]] = []
		
		print(f"Loading {split} dataset...")
		for img_id in tqdm(img_ids, desc=f"Loading {split} samples"):
			img_info = self.coco.loadImgs(img_id)[0]
			ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
			anns = self.coco.loadAnns(ann_ids)
			
			# Filter annotations
			for ann in anns:
				# Check if annotation has keypoints and at least one is labeled (v=1 or v=2)
				if 'keypoints' in ann and sum(1 for v in ann['keypoints'][2::3] if v > 0) > 0:
					# Check if bbox is valid
					bbox = ann['bbox']
					if bbox[2] > 0 and bbox[3] > 0:  # width and height > 0
						self.samples.append({
							'image_id': img_id,
							'image_path': os.path.join(coco_dir, split, img_info['file_name']),
							'bbox': bbox,  # [x, y, width, height]
							'keypoints': np.array(ann['keypoints']).reshape(-1, 3)  # [17, 3] (x, y, v)
						})
		
		# Limit dataset size if requested
		if max_samples is not None and max_samples < len(self.samples):
			self.samples = self.samples[:max_samples]
			
		print(f"Loaded {len(self.samples)} samples from COCO {split}")

	def _load_image(self, image_path):
		"""Load and preprocess image"""
		try:
			img = cv2.imread(image_path)
			if img is None:
				raise Exception(f"Failed to load image: {image_path}")
			return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		except Exception as e:
			print(f"Error loading image {image_path}: {e}")
			return None

	def _crop_person(
		self, 
		img: np.ndarray, 
		bbox: List[float], 
		margin: float = 0.2
	) -> Tuple[np.ndarray, float, float]:
		"""Crop person from image with margin"""
		x, y, w, h = bbox
		x1 = max(0, int(x - margin * w))
		y1 = max(0, int(y - margin * h))
		x2 = min(img.shape[1], int(x + w + margin * w))
		y2 = min(img.shape[0], int(y + h + margin * h))
		
		person_img = img[y1:y2, x1:x2]
		# Handle invalid crops
		if person_img.shape[0] == 0 or person_img.shape[1] == 0:
			print(f"Warning: Invalid crop for bbox {bbox} in image shape {img.shape}. Using full image.")
			person_img = img
			x1, y1 = 0, 0
		
		return person_img, x1, y1

	def _adjust_keypoints(
		self, 
		keypoints: np.ndarray, 
		x1: float, 
		y1: float
	) -> np.ndarray:
		"""Adjust keypoints coordinates after cropping"""
		adjusted_keypoints = keypoints.copy()
		# Only adjust labeled keypoints (v=1 or v=2)
		labeled_mask = adjusted_keypoints[:, 2] > 0
		adjusted_keypoints[labeled_mask, 0] -= x1
		adjusted_keypoints[labeled_mask, 1] -= y1
		return adjusted_keypoints

	def _apply_random_occlusion(
		self,
		img: np.ndarray,
		keypoints: np.ndarray # Keypoints scaled to input_size, BEFORE visibility normalization
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Apply random rectangular occlusions and update keypoint visibility."""
		img_h, img_w = img.shape[:2]
		num_occlusions = np.random.randint(1, self.max_occlusions + 1)
		output_img = img.copy()
		output_keypoints = keypoints.copy()

		for _ in range(num_occlusions):
			# Determine occlusion size
			occ_h_ratio = np.random.uniform(*self.occlusion_size_range)
			occ_w_ratio = np.random.uniform(*self.occlusion_size_range)
			occ_h = int(img_h * occ_h_ratio)
			occ_w = int(img_w * occ_w_ratio)

			# Determine occlusion position (top-left corner)
			if img_h > occ_h:
				occ_y = np.random.randint(0, img_h - occ_h)
			else:
				occ_y = 0 # Handle case where occlusion is image height
			if img_w > occ_w:
				occ_x = np.random.randint(0, img_w - occ_w)
			else:
				occ_x = 0 # Handle case where occlusion is image width

			# Draw occlusion (using black color for simplicity)
			output_img[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w, :] = 0

			# Update keypoint visibility if occluded
			for i in range(self.num_keypoints):
				# Check only originally visible keypoints (v=2)
				if output_keypoints[i, 2] == 2:
					kpt_x, kpt_y = output_keypoints[i, :2]
					# Check if the keypoint falls within the occlusion rectangle
					if (occ_x <= kpt_x < occ_x + occ_w) and \
					   (occ_y <= kpt_y < occ_y + occ_h):
						output_keypoints[i, 2] = 1 # Change visibility from visible (2) to occluded (1)

		return output_img, output_keypoints

	def _create_heatmap_offsets(
		self, 
		keypoints: np.ndarray
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Creates heatmaps, offset maps, and offset weights for keypoints.

		Args:
			keypoints (np.ndarray): Keypoints scaled to input_size [17, 3].

		Returns:
			Tuple[np.ndarray, np.ndarray, np.ndarray]:
				- heatmaps (np.ndarray): [num_kpts, hm_size, hm_size]
				- offsets (np.ndarray): [num_kpts * 2, hm_size, hm_size] (dx, dy order)
				- offset_weights (np.ndarray): [num_kpts * 2, hm_size, hm_size]
		"""
		hm_size = self.heatmap_size
		num_kpts = self.num_keypoints
		heatmaps = np.zeros((num_kpts, hm_size, hm_size), dtype=np.float32)
		offsets = np.zeros((num_kpts * 2, hm_size, hm_size), dtype=np.float32)
		offset_weights = np.zeros((num_kpts * 2, hm_size, hm_size), dtype=np.float32)

		# Generate coordinates for heatmap grid
		y_grid, x_grid = np.mgrid[0:hm_size, 0:hm_size]

		for i in range(num_kpts):
			# Process only labeled keypoints (visible or occluded)
			if keypoints[i, 2] > 0:
				# Scale keypoint coordinate to heatmap size
				x_hm, y_hm = keypoints[i, :2] * hm_size / self.input_size

				# Ensure keypoint is within bounds before drawing heatmap/offset
				if not (0 <= x_hm < hm_size and 0 <= y_hm < hm_size):
						# Mark the keypoint as unlabeled for the target tensor if it falls outside
						# keypoints[i, 2] = 0 # Optional: Or handle this in loss? Let's keep it for now.
						continue

				# --- Heatmap Generation ---
				d2 = (x_grid - x_hm)**2 + (y_grid - y_hm)**2
				# Prevent division by zero if sigma is extremely small
				exponent = d2 / (2 * self.sigma**2 + 1e-6)
				heatmaps[i] = np.exp(-exponent)

				# --- Offset Map Generation ---
				# Get integer coordinates
				ix, iy = int(x_hm), int(y_hm)

				# Define the region around the keypoint to fill offsets
				x_min = max(0, ix - self.offset_radius)
				y_min = max(0, iy - self.offset_radius)
				x_max = min(hm_size, ix + self.offset_radius + 1)
				y_max = min(hm_size, iy + self.offset_radius + 1)

				# Iterate through the region
				for py in range(y_min, y_max):
					for px in range(x_min, x_max):
						# Calculate exact offset from pixel center to GT keypoint
						dx = x_hm - px
						dy = y_hm - py

						# Check if within radius (optional, but standard)
						# if dx**2 + dy**2 <= self.offset_radius**2: # More accurate circle
						offsets[i * 2,     py, px] = dx
						offsets[i * 2 + 1, py, px] = dy
						offset_weights[i * 2,     py, px] = 1
						offset_weights[i * 2 + 1, py, px] = 1

		return heatmaps, offsets, offset_weights

	def _normalize_visibility(self, keypoints: np.ndarray) -> np.ndarray:
		"""Normalizes keypoint visibility based on the chosen handling method."""
		norm_keypoints = keypoints.copy()
		if self.visibility_handling == 'binary':
			# v=0 -> 0.0, v=1 -> 1.0, v=2 -> 1.0
			norm_keypoints[:, 2] = (norm_keypoints[:, 2] > 0).astype(float)
		elif self.visibility_handling == 'scaled':
			# v=0 -> 0.0, v=1 -> 0.5, v=2 -> 1.0
			norm_keypoints[:, 2] = norm_keypoints[:, 2] / 2.0
		return norm_keypoints

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		sample = self.samples[idx]

		# Load and preprocess image
		img = self._load_image(sample['image_path'])

		# Get bounding box and keypoints
		bbox = sample['bbox']
		# Use original keypoints for processing, normalize visibility at the end
		keypoints_orig_vis = sample['keypoints'].copy() # Keep original v=0,1,2 for logic

		# Crop person with margin
		person_img, x1, y1 = self._crop_person(img, bbox)

		# Adjust keypoints to cropped image
		keypoints = self._adjust_keypoints(keypoints_orig_vis, x1, y1)

		# Data augmentation for training
		if self.augment and self.transform:
			keypoints_xy = [kp[:2] for kp in keypoints]
			visibility = keypoints[:, 2].copy()

			try:
				transformed = self.transform(image=person_img, keypoints=keypoints_xy)
				person_img = transformed['image']
				transformed_keypoints_xy = transformed['keypoints']
				
				# Reconstruct keypoints with original visibility
				if len(transformed_keypoints_xy) == self.num_keypoints:
					keypoints = np.array([[k[0], k[1], v] for k, v in zip(transformed_keypoints_xy, visibility)], dtype=np.float32)
				else:
					# Handle cases where keypoints might be removed by augmentation
					print(f"Warning: Keypoint number mismatch after augmentation ({len(transformed_keypoints_xy)} vs {self.num_keypoints}). Reconstructing visibility.")
					original_indices = [i for i, kp in enumerate(keypoints_xy) if kp in transformed_keypoints_xy]
					new_keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
					transformed_idx = 0
					for i in range(self.num_keypoints):
						if i in original_indices:
							k = transformed_keypoints_xy[transformed_idx]
							new_keypoints[i] = [k[0], k[1], visibility[i]]
							transformed_idx += 1
						else:
							new_keypoints[i] = [0, 0, 0] # Mark as unlabeled
					keypoints = new_keypoints
					
			except Exception as e:
				print(f"Error during Albumentations transform: {e}. Skipping augmentation for this sample.")

		# Resize image to input size AFTER geometric augmentations
		h_orig, w_orig = person_img.shape[:2]
		person_img_resized = cv2.resize(person_img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

		# Scale keypoints to input size (using potentially augmented keypoints)
		keypoints_scaled = keypoints.copy().astype(np.float32) 

		labeled_mask = keypoints_scaled[:, 2] > 0
		scale_x = self.input_size / w_orig if w_orig > 0 else 1
		scale_y = self.input_size / h_orig if h_orig > 0 else 1

		keypoints_scaled[labeled_mask, 0] *= scale_x
		keypoints_scaled[labeled_mask, 1] *= scale_y

		# --- Apply Random Occlusion --- 
		if self.augment and np.random.rand() < self.occlusion_prob:
			person_img_resized, keypoints_scaled = self._apply_random_occlusion(person_img_resized, keypoints_scaled)

		# --- Generate Heatmaps and Offsets using potentially occluded keypoints_scaled --- 
		# Pass keypoints_scaled which has coordinates relative to input_size
		heatmaps, offsets, offset_weights = self._create_heatmap_offsets(keypoints_scaled)

		# --- Normalize visibility for the target keypoints tensor ---
		keypoints_final_target = self._normalize_visibility(keypoints_scaled)

		# --- Combine heatmaps and offsets for model output target ---
		# Target shape: [num_kpts * 3, hm_size, hm_size]
		# Order: hm0, off0_x, off0_y, hm1, off1_x, off1_y, ...
		combined_hm_off = np.zeros((self.num_keypoints * 3, self.heatmap_size, self.heatmap_size), dtype=np.float32)
		for i in range(self.num_keypoints):
				combined_hm_off[i * 3, :, :] = heatmaps[i]
				combined_hm_off[i * 3 + 1, :, :] = offsets[i * 2]     # dx
				combined_hm_off[i * 3 + 2, :, :] = offsets[i * 2 + 1] # dy

		# Convert to tensors
		img_tensor = torch.from_numpy(person_img_resized.transpose(2, 0, 1)).float() / 255.0
		keypoints_tensor = torch.from_numpy(keypoints_final_target).float() # Target keypoints (x, y, vis) in input_size scale
		combined_hm_off_tensor = torch.from_numpy(combined_hm_off).float()  # Target heatmaps+offsets
		offset_weights_tensor = torch.from_numpy(offset_weights).float()    # Weights for offset loss

		return {
			'image': img_tensor,
			'keypoints': keypoints_tensor, # Target for regression branch [N, 17, 3]
			'combined_heatmap_offsets': combined_hm_off_tensor, # Target for heatmap branch [N, 51, H/4, W/4]
			'offset_weights': offset_weights_tensor, # Loss weights for offsets [N, 34, H/4, W/4]
			'image_id': sample['image_id']
		}

# --- Visualization (Needs update for new return dict) ---
if __name__ == '__main__':
	sns.set_style("whitegrid")
	sns.set_context("notebook", font_scale=1.2)

	# Initialize dataset
	coco_dir = "data/coco_keypoints" # Make sure this path is correct
	if not os.path.exists(coco_dir):
		print(f"Error: COCO directory not found at {coco_dir}")
		print("Please download the COCO dataset (images and annotations) and update the path.")
		exit()

	# Create dataset for visualization/statistics
	dataset = COCOKeypointsDataset(
		coco_dir=coco_dir,
		split='train2017',
		input_size=256,
		max_samples=None,
		augment=True,
		visibility_handling='scaled'
	)

	if len(dataset) == 0:
		print("Dataset loaded 0 samples. Check COCO path and annotation files.")
		exit()

	print(f"\nDataset Statistics:")
	print(f"Training set: {len(dataset)} samples")
	print(f"Total samples: {len(dataset)} samples")

	dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

	# Define keypoint names for better visualization
	keypoint_names = [
		'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
		'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
		'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
		'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
	]
	num_kpts = dataset.num_keypoints

	# Visualize first N samples
	num_visualize = 5
	print(f"\nVisualizing first {num_visualize} samples...")
	for i, sample_batch in enumerate(dataloader):
		if i >= num_visualize:
			break
		print(f"\n--- Visualizing Sample {i+1} ---")
		visualize_sample(sample_batch, keypoint_names, num_kpts)