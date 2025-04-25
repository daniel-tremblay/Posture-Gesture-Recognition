import os
import cv2
import math
import json
import torch
import random
import argparse
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import plot_hand_data, transform_bbox_to_crop

def crop_hand(
  image,
  bbox,
  output_size=256, # 256x256
  padding_factor=0.1
):
	"""
	Crop the hand from the selected image, adding padding around the bounding box.

	Steps:
	- Get original bounding box coordinates [x1, y1, x2, y2].
	- Calculate the center (cx, cy) and size (w_orig, h_orig) of the original bbox.
	- Determine the side length of the square crop region based on the max dimension and padding_factor.
	- Calculate the padded bounding box coordinates [x1_pad, y1_pad, x2_pad, y2_pad].
	- Clamp padded coordinates to image boundaries to get final crop coordinates [x1_final, y1_final, x2_final, y2_final].
	- Calculate final crop width and height.
	- Handle zero-size crops.
	- Crop the image using the final bounding box coordinates.
	- Resize the crop to output_size.
	- Calculate offset (final crop coords) and scale factors based on the final crop dimensions.

	Args:
		image (np.array): Input image in RGB format with shape (H, W, 3).
		bbox (list): Bounding box in absolute pixel coordinates [x1, y1, x2, y2].
		output_size (int): Target crop size (default is 256).
		padding_factor (float): Factor to expand the bounding box size (e.g., 0.1 for 10% padding on each side).

	Returns:
		crop_resized (np.array): Cropped and resized image of shape (output_size, output_size, 3).
		offset (tuple): (left, top, right, bottom) coordinates of the final crop in the original image.
		scale (tuple): (scale_x, scale_y) scaling factors applied when resizing the crop.
	"""
	h, w = image.shape[:2]

	# Get original bbox coordinates
	x1_orig, y1_orig, x2_orig, y2_orig = map(int, map(round, bbox))

	# Calculate original bbox center and dimensions
	cx = (x1_orig + x2_orig) / 2
	cy = (y1_orig + y2_orig) / 2
	w_orig = x2_orig - x1_orig
	h_orig = y2_orig - y1_orig

	# Handle zero-size original bbox before padding
	if w_orig <= 0 or h_orig <= 0:
		print(f"Warning: Zero-size original bounding box encountered: {bbox}. Skipping sample.")
		return None, None, None

	# Determine the side length of the square crop region
	max_dim = max(w_orig, h_orig)
	side_len = max_dim * (1 + 2 * padding_factor) # Expand by padding_factor on each side

	# Calculate padded bounding box coordinates (square centered at original bbox center)
	x1_pad = int(round(cx - side_len / 2))
	y1_pad = int(round(cy - side_len / 2))
	x2_pad = int(round(cx + side_len / 2))
	y2_pad = int(round(cy + side_len / 2))

	# Clamp padded coordinates to image boundaries to get the final crop coordinates
	x1_final = max(0, x1_pad)
	y1_final = max(0, y1_pad)
	x2_final = min(w, x2_pad)
	y2_final = min(h, y2_pad)

	# Calculate final crop width and height
	crop_w_final = x2_final - x1_final
	crop_h_final = y2_final - y1_final

	# Handle potential zero-size crop after clamping
	if crop_w_final <= 0 or crop_h_final <= 0:
		print(f"Warning: Zero-size final crop region after padding and clamping: {[x1_final, y1_final, x2_final, y2_final]} from original {bbox} in image of size {(h, w)}. Skipping sample.")
		return None, None, None

	# Crop the image using the final bounding box coordinates
	crop = image[y1_final:y2_final, x1_final:x2_final]

	# Resize the crop to output_size
	crop_resized = cv2.resize(crop, (output_size, output_size))

	# Calculate scaling factors based on the final crop dimensions
	scale_x = output_size / crop_w_final
	scale_y = output_size / crop_h_final

	# Offset is the coordinates of the final crop in the original image
	offset = (x1_final, y1_final, x2_final, y2_final)

	return crop_resized, offset, (scale_x, scale_y)

def collate_filter_none(batch):
	"""Collate function that filters out None items from the batch."""
	batch = list(filter(lambda x: x is not None and all(item is not None for item in x), batch))

	if not batch:
		return None
    
	images = torch.stack([item[0] for item in batch])
	landmarks = torch.stack([item[1] for item in batch])
	return images, landmarks
  
class HagridDataset(Dataset):
	"""
	PyTorch Dataset for training a MediaPipe Hands landmarks estimating model.

	- This Dataset class processes hand gesture annotations stored in JSON format (organized under
	annotations/<split>) along with corresponding images stored under images/<gesture>.
	- Each hand instance in an image is treated as a separate sample.
	- The cropping strategy uses the provided bounding box to extract a crop that is then resized to 256x256 (see crop_hand()).
	- The 21 hand landmarks are adjusted accordingly and normalized relative to the crop
	- Augmentations like horizontal flip, brightness/contrast adjustments, and grayscale conversion can be applied.

	A sample consists of:
	  - crop_img: A tensor of shape [3, 256, 256] with image pixel values in [0, 1].
		- landmarks: A tensor of shape [21, 2] with normalized landmark coordinates in [0, 1].
	"""
	def __init__(
    self,
    root,
    split="train",
    output_size=256,
    augment=False,
    flip_prob=0.5,
    grayscale_prob=0.3,
    brightness_contrast_prob=0.4,
    color_jitter_prob=0.4,
    vflip_prob=0.1,
    rotate_prob=0.1,
    blur_prob=0.0,
    crop_padding_factor=1.0
	):
		"""
		Args:
			root (str): Root directory of the dataset
			split (str): One of {"train", "test", "val"}.
			output_size (int): The target size for the cropped image (default 256).
			augment (bool): Whether to apply data augmentation (default False).
			flip_prob (float): Probability of applying horizontal flip augmentation.
			vflip_prob (float): Probability of applying vertical flip augmentation.
			grayscale_prob (float): Probability of applying grayscale augmentation.
			brightness_contrast_prob (float): Probability of applying brightness/contrast augmentation.
			color_jitter_prob (float): Probability of applying color jitter augmentation.
			rotate_prob (float): Probability of applying slight rotation augmentation.
			blur_prob (float): Probability of applying Gaussian blur augmentation.
			crop_padding_factor (float): Factor to expand bbox for cropping.
		"""
		self.root = root
		self.split = split
		self.output_size = output_size
		self.augment = augment
		self.blur_prob = blur_prob
		self.crop_padding_factor = crop_padding_factor

		# --- Augmentations ---
		self.transform = None
		if self.augment:
			self.transform = A.Compose([
				A.HorizontalFlip(p=flip_prob),
				A.VerticalFlip(p=vflip_prob),
				A.RandomBrightnessContrast(p=brightness_contrast_prob),
				A.ColorJitter(p=color_jitter_prob),
				A.ToGray(p=grayscale_prob),
				A.Rotate(limit=(-10, 10), p=rotate_prob, border_mode=cv2.BORDER_CONSTANT),
				A.GaussianBlur(p=self.blur_prob),
			], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

		# Directories for annotations and images.
		self.annotations_dir = os.path.join(root, "annotations", split)
		self.images_dir = os.path.join(root, "images")

		# Build the sample list.
		self.samples = []
		json_files = glob(os.path.join(self.annotations_dir, "*.json"))
		for file in json_files:
			gesture = os.path.splitext(os.path.basename(file))[0]
			with open(file, "r") as f:
				data = json.load(f)

			for ann_id, ann in data.items():
        # --- Validate samples ---
				# Only process annotations with non-empty hand_landmarks.
				if "hand_landmarks" not in ann or not ann["hand_landmarks"]:
					continue
				# Check required keys exist and have matching lengths.
				if not ("bboxes" in ann and "hand_landmarks" in ann and "labels" in ann):
					continue
				if not (len(ann["bboxes"]) == len(ann["hand_landmarks"]) == len(ann["labels"])):
					continue

        # --- Process samples ---
				# For each hand in the annotation, treat it as a separate sample.
				for idx in range(len(ann["hand_landmarks"])):
					# Ensure the hand landmarks array has exactly 21 points.
					if len(ann["hand_landmarks"][idx]) != 21:
						continue

					sample = {
            "ann_id": ann_id,
						"image_path": os.path.join(self.images_dir, gesture, f"{ann_id}.jpg"),
						"gesture": gesture,
						"bbox_rel": ann["bboxes"][idx], # COCO format [x, y, w, h] (relative to image)
						"landmarks_rel": ann["hand_landmarks"][idx] # List of 21 [x, y] points
					}
					self.samples.append(sample)

	def __len__(self):
		return len(self.samples)

	def _adjust_landmarks_to_crop(
		self, 
		landmarks_rel, 
		original_image_shape, 
		crop_coords, 
		scales
	):
		"""
		Adjusts landmarks from original image coordinates to cropped/resized image coordinates.

		Args:
			landmarks_rel (list): List of [x, y] relative landmark coordinates (in [0, 1]) from the original image.
			original_image_shape (tuple): The (height, width) of the original image.
			crop_coords (tuple): The absolute coordinates (left, top, right, bottom) of the crop in the original image.
			scales (tuple): The scaling factors (scale_x, scale_y) applied when resizing the crop.

		Returns:
			np.ndarray: Adjusted landmarks as a NumPy array of shape (21, 2), normalized to [0, 1] relative to the crop.
		"""
		h_orig, w_orig = original_image_shape
		offset_x, offset_y, _, _ = crop_coords
		scale_x, scale_y = scales

		landmarks_cropped = []
		for point in landmarks_rel:
			# Convert relative landmark to absolute coordinates in the original image.
			orig_x = point[0] * w_orig
			orig_y = point[1] * h_orig
			
			# Transform to relative coordinates within the resized crop, normalized to [0, 1].
			# 1. Shift by the crop offset.
			# 2. Multiply by the appropriate scaling factor (scale_x or scale_y).
			# 3. Normalize by dividing by output_size.
			new_x = (orig_x - offset_x) * scale_x / self.output_size
			new_y = (orig_y - offset_y) * scale_y / self.output_size
			landmarks_cropped.append([new_x, new_y])
			
		return np.array(landmarks_cropped, dtype=np.float32)

	def __getitem__(self, idx):
		try:
			sample = self.samples[idx]
			image_path = sample["image_path"]
			image = cv2.imread(image_path)
			if image is None:
				print(f"Warning: Image not found or unreadable: {image_path}. Skipping sample {idx}.")
				return None, None, None, None, None

			# Convert image from BGR to RGB.
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			h, w, _ = image.shape

			# Convert relative bounding box [x, y, w, h] to absolute coordinates.
			x_rel, y_rel, bw_rel, bh_rel = sample["bbox_rel"]
			x1 = x_rel * w
			y1 = y_rel * h
			x2 = (x_rel + bw_rel) * w
			y2 = (y_rel + bh_rel) * h
			bbox_abs = [x1, y1, x2, y2]

			# Crop the image with padding
			crop_img, crop_coords, scales = crop_hand(
				image,
				bbox_abs,
				output_size=self.output_size,
				padding_factor=self.crop_padding_factor
			)

			# Check if cropping failed
			if crop_img is None:
				print(f"Warning: Cropping failed for sample {idx} ({image_path}). Skipping.")
				return None, None, None, None, None

			# Normalize landmarks to the cropped image space [0, 1]
			landmarks_cropped_normalized = self._adjust_landmarks_to_crop(
				landmarks_rel=sample["landmarks_rel"],
				original_image_shape=(h, w),
				crop_coords=crop_coords,
				scales=scales
			)

			# Convert normalized landmarks to absolute pixel coordinates within the crop for augmentation
			landmarks_abs_crop = (landmarks_cropped_normalized * self.output_size).tolist()

			# Apply augmentations if enabled
			if self.augment and self.transform:
				transformed = self.transform(image=crop_img, keypoints=landmarks_abs_crop)
				crop_img = transformed['image']
				landmarks_abs_crop = np.array(transformed['keypoints'], dtype=np.float32)
				landmarks_cropped_normalized = landmarks_abs_crop / self.output_size
			else:
				landmarks_cropped_normalized = np.array(landmarks_cropped_normalized, dtype=np.float32)

			# Convert image to float32, normalize to [0, 1], convert to CHW tensor.
			crop_img = crop_img.astype(np.float32) / 255.0
			crop_img = torch.from_numpy(crop_img).permute(2, 0, 1)

			# Convert landmarks to tensor.
			landmarks_cropped = torch.from_numpy(landmarks_cropped_normalized)

			# Return all necessary info
			return crop_img, landmarks_cropped, bbox_abs, crop_coords, scales

		except Exception as e:
			print(f"Error processing sample {idx} ({sample.get('image_path', 'unknown path')}): {e}. Skipping.")
			return None, None, None, None, None

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Load HagridDataset and display sample info.")
	parser.add_argument(
		"--dir",
		type=str,
		default="data/hagrid_small",
		help="Root directory of the dataset (containing annotations/ and images/)"
	)
	parser.add_argument(
		"--split",
		type=str,
		default="train",
		choices=["train", "val", "test"],
		help="Dataset split to load (train, val, test)"
	)
	parser.add_argument(
		"--augment",
		action="store_true",
		help="Apply data augmentation when loading samples for visualization"
	)
	parser.add_argument(
		"--flip_prob",
		type=float,
		default=0.5,
		help="Probability of horizontal flip if --augment is used."
	)
	parser.add_argument(
		"--vflip_prob",
		type=float,
		default=0.1,
		help="Probability of vertical flip if --augment is used."
	)
	parser.add_argument(
		"--color_jitter_prob",
		type=float,
		default=0.4,
		help="Probability of color jitter if --augment is used."
	)
	parser.add_argument(
		"--rotate_prob",
		type=float,
		default=0.1,
		help="Probability of slight rotation if --augment is used."
	)
	parser.add_argument(
		"--blur_prob",
		type=float,
		default=0.0,
		help="Probability of Gaussian blur if --augment is used."
	)
	parser.add_argument(
		"--crop_padding_factor",
		type=float,
		default=1,
		help="Factor to pad the bounding box before cropping (e.g., 0.1 for 10%%)."
	)
	parser.add_argument(
		"--labels",
		action="store_true",
		help="Show the landmark index labels on the visualization."
	)
	args = parser.parse_args()
	
	print("\n")
	print(f"Dataset root: {args.dir}")
	print(f"Split: {args.split}")
	print(f"Augmentations enabled: {args.augment}")
	print(f"Show labels: {args.labels}")
	print(f"Crop padding factor: {args.crop_padding_factor}")
	if args.augment:
		print("\n")
		print("--- Augmentation Arguments ---")
		print(f"Horizontal flip probability: {args.flip_prob}")
		print(f"Vertical flip probability: {args.vflip_prob}")
		print(f"Color jitter probability: {args.color_jitter_prob}")
		print(f"Rotate probability: {args.rotate_prob}")
		print(f"Blur probability: {args.blur_prob}")
	print("\n" + "-"*25 + "\n")

	dataset_root = args.dir
	split = args.split
	output_size = 256
	
	# Load the dataset
	try:
		dataset = HagridDataset(
			root=dataset_root,
			split=split,
			output_size=output_size,
			augment=args.augment,
			flip_prob=args.flip_prob,
			vflip_prob=args.vflip_prob,
			color_jitter_prob=args.color_jitter_prob,
			rotate_prob=args.rotate_prob,
			blur_prob=args.blur_prob,
			crop_padding_factor=args.crop_padding_factor
		)
		dataset_size = len(dataset)
		print(f"Number of {split} samples: {dataset_size}")

		if dataset_size > 0:
			sample_idx = random.randint(0, dataset_size - 1)
			
			result = None
			try:
				result = dataset[sample_idx]
			except Exception as e:
				print(f"Error loading index {sample_idx}: {e}")
				result = None

			if result is None or any(item is None for item in result):
				print(f"Failed to load a valid sample from index {sample_idx}. It might be problematic or __getitem__ returned None.")
			else:
				print(f"Successfully loaded sample index: {sample_idx}")
				cropped_image_tensor, cropped_image_landmarks, bbox_abs, crop_coords, scales = result
				offset_x, offset_y, _, _ = crop_coords
				scale_x, scale_y = scales
				x1_abs, y1_abs, x2_abs, y2_abs = bbox_abs

				print("Sample cropped image tensor shape:", cropped_image_tensor.shape)
				print("Sample landmarks shape:", cropped_image_landmarks.shape)

				selected_sample_info = dataset.samples[sample_idx]
				original_image_path = selected_sample_info['image_path']
				gesture_name = selected_sample_info['gesture']
				ann_id = selected_sample_info['ann_id']

				original_image = cv2.imread(original_image_path)
				if original_image is not None:
					original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
					h_orig, w_orig, _ = original_image_rgb.shape

					cropped_image_np = cropped_image_tensor.permute(1, 2, 0).numpy()
					cropped_image_np = (cropped_image_np * 255).astype(np.uint8)
					h_crop, w_crop, _ = cropped_image_np.shape

					# Calculate bbox coordinates in the resized crop
					bbox_crop_xyxy = transform_bbox_to_crop(
						bbox_abs=bbox_abs,
						crop_coords=crop_coords,
						scales=scales,
						crop_size=(h_crop, w_crop)
					)

					# Convert normalized landmarks back to absolute pixel coordinates in the crop
					landmarks_plot = cropped_image_landmarks.numpy() * output_size

					# --- Plot the original and cropped images with landmarks ---
					titles = {
						'title': f"Annotation ID: {ann_id}, Index: {sample_idx}",
						'original': f"Original Image\n({w_orig}x{h_orig})",
						'cropped': f"Cropped Image\nGesture: {gesture_name}\n({w_crop}x{h_crop})"
					}

					plot_hand_data(
						original_image=original_image_rgb,
						cropped_image=cropped_image_np,
						landmarks_crop=landmarks_plot,
						bbox_original=bbox_abs,
						bbox_crop=bbox_crop_xyxy,
						titles=titles,
						landmark_size=80,
						show_labels=args.labels
					)
				else:
					print(f"Error: Could not load original image at {original_image_path}")

		else:
			print("Dataset is empty. Cannot retrieve sample[0].")
			print(f"Please check if JSON files exist in: {os.path.abspath(os.path.join(dataset_root, 'annotations', split))}")
	except FileNotFoundError:
		print(f"Error: Annotation directory not found at {os.path.abspath(os.path.join(dataset_root, 'annotations', split))}")
	except Exception as e:
		print(f"An unexpected error occurred while loading or plotting: {e}")
