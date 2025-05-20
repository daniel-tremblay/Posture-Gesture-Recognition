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
  padding_factor=1.0
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
	"""Filters out None items and stacks images, landmarks, and labels."""
	original_len = len(batch)
	batch = list(filter(lambda x: x is not None and all(item is not None for item in x), batch))
	filtered_len = len(batch)

	if filtered_len < original_len:
		print(f"Collate Warning: Filtered out {original_len - filtered_len} samples due to None values.")

	if not batch:
		print("Collate Warning: Batch is empty after filtering.")
		return None

	try:
		images = torch.stack([item[0] for item in batch])
		landmarks = torch.stack([item[1] for item in batch])
		labels = torch.stack([item[2] for item in batch])
		return images, landmarks, labels
		
	except Exception as e:
		print(f"Collate Error: Failed to stack batch components. Error: {e}")
		return None

class HagridDataset(Dataset):
	"""
	Dataset Class for the Hagrid dataset, providing cropped images, hand landmarks, and gesture classification labels.

	- This Dataset class processes hand gesture annotations stored in JSON format (organized under 
	annotations/<split>) along with corresponding images stored under images/<gesture>.
	- Each hand instance is a separate sample.
	- Crops hand using bbox, resizes to output_size.
	- Adjusts landmarks relative to the crop.
	- Provides gesture class index.
	- Applies augmentations if enabled.

	Output:
	  - crop_img: Tensor [C, H, W] (e.g., [3, 256, 256]), float32, normalized to [0, 1].
	  - landmarks: Tensor [21, 2], float32, normalized to [0, 1] relative to the crop.
	  - label_index: Tensor [], int64, the gesture class index.
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
		vflip_prob=0.0,
		rotate_prob=0.1,
		blur_prob=0.0,
		crop_padding_factor=1.0,
		ignore_classes=None,
		rotate_limit=5
	):
		"""
		Args:
			root (str): Root directory of the dataset.
			split (str): One of {"train", "val", "test"}.
			output_size (int): Target size for cropped image (default 256).
			augment (bool): Apply data augmentation (default False).
			flip_prob (float): Probability of horizontal flip.
			grayscale_prob (float): Probability of grayscale conversion.
			brightness_contrast_prob (float): Probability of brightness/contrast adjustment.
			color_jitter_prob (float): Probability of color jitter.
			vflip_prob (float): Probability of vertical flip.
			rotate_prob (float): Probability of rotation.
			blur_prob (float): Probability of Gaussian blur.
			crop_padding_factor (float): Factor to pad bbox for cropping.
			ignore_classes (list[str], optional): List of gesture names to ignore. Defaults to None.
		"""
		self.root = root
		self.split = split
		self.output_size = output_size
		self.augment = augment
		self.blur_prob = blur_prob
		self.crop_padding_factor = crop_padding_factor
		self.ignore_classes = set(ignore_classes) if ignore_classes else set()

		self.annotations_dir = os.path.join(root, "annotations", split)
		self.images_dir = os.path.join(root, "images")

		# --- Augmentations ---
		keypoint_params = A.KeypointParams(format='xy', remove_invisible=False) if self.augment else None

		self.transform = None
		if self.augment:
			self.transform = A.Compose([
				A.HorizontalFlip(p=flip_prob),
				A.VerticalFlip(p=vflip_prob),
				A.RandomBrightnessContrast(p=brightness_contrast_prob),
				A.ColorJitter(p=color_jitter_prob),
				A.ToGray(p=grayscale_prob),
				A.Rotate(limit=(-rotate_limit, rotate_limit), p=rotate_prob, border_mode=cv2.BORDER_CONSTANT),
				A.GaussianBlur(p=self.blur_prob),
			], keypoint_params=keypoint_params)

		# --- Create Label Mappings---
		self.label_to_index = {}
		self.index_to_label = {}
		self.num_classes = 0
		all_gestures = set()
		json_labels = glob(os.path.join(self.annotations_dir, "*.json"))
		
		ignored_gestures_found = set()

		if not json_labels:
			print(f"Warning: No JSON files found in {self.annotations_dir}. Cannot create label map.")
		else:
			for file in json_labels:
				gesture_name = os.path.splitext(os.path.basename(file))[0]
				if gesture_name in self.ignore_classes:
					ignored_gestures_found.add(gesture_name)
					continue
				all_gestures.add(gesture_name)
			
			if self.ignore_classes:
				print(f"Ignoring {len(ignored_gestures_found)} gestures specified in ignore_classes: {', '.join(sorted(list(ignored_gestures_found)))}")

			sorted_gestures = sorted(list(all_gestures))
			self.label_to_index = {name: i for i, name in enumerate(sorted_gestures)}
			self.index_to_label = {i: name for i, name in enumerate(sorted_gestures)}
			self.num_classes = len(sorted_gestures)
			print(f"Found {self.num_classes} unique gestures in '{split}' split.")

		# --- Build the sample list ---
		self.samples = []
		json_files = glob(os.path.join(self.annotations_dir, "*.json"))
		skipped_samples = 0
		required_keys = ["bboxes", "labels", "hand_landmarks"]

		for file in json_files:
			gesture = os.path.splitext(os.path.basename(file))[0]
			
			if gesture in self.ignore_classes:
				continue

			try:
				with open(file, "r") as f:
					data = json.load(f)
			except json.JSONDecodeError:
				print(f"Warning: Could not decode JSON file: {file}. Skipping.")
				continue
			except FileNotFoundError:
				print(f"Warning: Annotation file not found: {file}. Skipping")
				continue

			for ann_id, ann in data.items():
				# --- Validate samples ---
				# Check if all required keys are present
				if not all(key in ann for key in required_keys):
					skipped_samples += 1
					continue

				# Check for non-empty lists for all required fields
				if not all(ann[key] for key in required_keys):
					skipped_samples += 1
					continue

				# Check for matching lengths
				if not (len(ann["bboxes"]) == len(ann["hand_landmarks"]) == len(ann["labels"])):
					skipped_samples += 1
					continue

				# --- Process samples ---
				num_hands = len(ann["bboxes"])
				for idx in range(num_hands):
					if len(ann["hand_landmarks"][idx]) != 21:
						skipped_samples += 1
						continue

					# Check if gesture label exists in the map
					if gesture not in self.label_to_index:
						print(f"Warning: Gesture '{gesture}' from file {file} not found in label map. Skipping sample {ann_id}, index {idx}.")
						skipped_samples += 1
						continue

					landmarks_rel = ann["hand_landmarks"][idx] # List of 21 [x, y] points
					bbox_rel = ann["bboxes"][idx] # COCO format [x, y, w, h] (relative to image)
					sample = {
						"ann_id": ann_id,
						"image_path": os.path.join(self.images_dir, gesture, f"{ann_id}.jpg"),
						"gesture": gesture,
						"bbox_rel": bbox_rel,
						"landmarks_rel": landmarks_rel
					}
					self.samples.append(sample)

		if skipped_samples > 0:
			print(f"Skipped {skipped_samples} annotations")
		if len(self.samples) == 0:
			print(f"Warning: Loaded 0 valid samples for split '{self.split}'. Please check annotations and data structure.")
		else:
			print(f"Loaded {len(self.samples)} valid samples for split '{self.split}'.")

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
		if landmarks_rel is None:
			return None

		h_orig, w_orig = original_image_shape
		offset_x, offset_y, _, _ = crop_coords
		scale_x, scale_y = scales

		landmarks_cropped = []
		valid = True
		for point in landmarks_rel:
			if not (isinstance(point, (list, tuple)) and len(point) == 2):
				valid = False
				break
			try:
				# Convert relative landmark to absolute coordinates in the original image.
				orig_x = float(point[0]) * w_orig
				orig_y = float(point[1]) * h_orig

				# Transform to relative coordinates within the resized crop, normalized to [0, 1].
				new_x = (orig_x - offset_x) * scale_x / self.output_size
				new_y = (orig_y - offset_y) * scale_y / self.output_size
				landmarks_cropped.append([new_x, new_y])
			except (TypeError, ValueError) as e:
				valid = False
				break

		if not valid:
			return None

		return np.array(landmarks_cropped, dtype=np.float32)

	def __getitem__(self, idx):
		"""
		Returns:
			tuple: (image_tensor, landmarks_tensor, label_index_tensor)
		"""
		image_tensor, landmarks_tensor, label_tensor = None, None, None # Initialize
		try:
			sample = self.samples[idx]
			image_path = sample["image_path"]
			gesture_label = sample["gesture"]
			landmarks_rel = sample["landmarks_rel"]
			bbox_rel = sample["bbox_rel"]

			# --- Load Image ---
			image = cv2.imread(image_path)
			if image is None:
				return None, None, None

			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			h, w, _ = image.shape

			# --- BBox Conversion ---
			# Convert relative bbox to absolute coordinates
			x_rel, y_rel, bw_rel, bh_rel = bbox_rel
			x1 = x_rel * w
			y1 = y_rel * h
			x2 = (x_rel + bw_rel) * w
			y2 = (y_rel + bh_rel) * h
			bbox_abs = [x1, y1, x2, y2]

			# --- Crop Image ---
			crop_img, crop_coords, scales = crop_hand(
				image,
				bbox_abs,
				output_size=self.output_size,
				padding_factor=self.crop_padding_factor
			)
			if crop_img is None:
				return None, None, None

			# --- Process Landmarks ---
			landmarks_cropped_normalized = self._adjust_landmarks_to_crop(
				landmarks_rel=landmarks_rel,
				original_image_shape=(h, w),
				crop_coords=crop_coords,
				scales=scales
			)
			
			landmarks_abs_crop = None
			if landmarks_cropped_normalized is not None:
				if landmarks_cropped_normalized.shape == (21, 2):
					landmarks_abs_crop = (landmarks_cropped_normalized * self.output_size).tolist()
				else:
					landmarks_cropped_normalized = None

			# --- Process Label ---
			label_index = self.label_to_index[gesture_label]
			label_tensor = torch.tensor(label_index, dtype=torch.long)

			# --- Apply Augmentations ---
			if self.augment and self.transform:
				aug_input = {'image': crop_img}
				if landmarks_abs_crop is not None:
					if isinstance(landmarks_abs_crop, np.ndarray):
						landmarks_abs_crop = landmarks_abs_crop.tolist()
					aug_input['keypoints'] = landmarks_abs_crop

				transformed = self.transform(**aug_input)
				crop_img = transformed['image']

				if 'keypoints' in transformed and landmarks_cropped_normalized is not None:
					landmarks_abs_crop_aug = np.array(transformed['keypoints'], dtype=np.float32)
					if landmarks_abs_crop_aug.shape == (21, 2):
						landmarks_cropped_normalized = landmarks_abs_crop_aug / self.output_size
					else:
						landmarks_cropped_normalized = None

			# --- Final Tensor Conversions ---
			# Image: HWC uint8 -> CHW float32 [0, 1]
			image_tensor = torch.from_numpy(crop_img.astype(np.float32) / 255.0).permute(2, 0, 1)

			# Landmarks: np.array [21, 2] or None -> Tensor [21, 2] float32 or None	
			if landmarks_cropped_normalized is not None:
				if landmarks_cropped_normalized.shape == (21, 2):
					landmarks_tensor = torch.from_numpy(landmarks_cropped_normalized.astype(np.float32))
				else:
					landmarks_tensor = None

			return image_tensor, landmarks_tensor, label_tensor

		except Exception as e:
			print(f"Error processing sample {idx} ({sample.get('image_path', 'unknown path')}): {e}")
			return None, None, None


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
		"--show_labels",
		action="store_true",
		help="Show the landmark index labels on the visualization."
	)
	parser.add_argument("--flip_prob", type=float, default=0.5)
	parser.add_argument("--vflip_prob", type=float, default=0.0)
	parser.add_argument("--grayscale_prob", type=float, default=0.1)
	parser.add_argument("--brightness_contrast_prob", type=float, default=0.2)
	parser.add_argument("--color_jitter_prob", type=float, default=0.2)
	parser.add_argument("--rotate_prob", type=float, default=0.1)
	parser.add_argument("--blur_prob", type=float, default=0.0)
	parser.add_argument("--crop_padding_factor", type=float, default=1.0)

	args = parser.parse_args()

	print("\n--- Dataset Configuration ---")
	print(f"Dataset root: {args.dir}")
	print(f"Split: {args.split}")
	print(f"Augmentations enabled: {args.augment}")
	print(f"Show landmark labels: {args.show_labels}")
	print(f"Crop padding factor: {args.crop_padding_factor}")
	
	if args.augment:
		print("\n--- Augmentation Parameters ---")
		print(f"HFlip: {args.flip_prob}, VFlip: {args.vflip_prob}, Gray: {args.grayscale_prob}")
		print(f"Bright/Contrast: {args.brightness_contrast_prob}, ColorJitter: {args.color_jitter_prob}")
		print(f"Rotate: {args.rotate_prob}, Blur: {args.blur_prob}")
	print("-" * 30 + "\n")

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
			grayscale_prob=args.grayscale_prob,
			brightness_contrast_prob=args.brightness_contrast_prob,
			color_jitter_prob=args.color_jitter_prob,
			rotate_prob=args.rotate_prob,
			blur_prob=args.blur_prob,
			crop_padding_factor=args.crop_padding_factor
		)
		dataset_size = len(dataset)
		print(f"Initialized dataset for split '{split}'. Number of valid samples: {dataset_size}")

		if dataset_size > 0:
			sample_idx = random.randint(0, dataset_size - 1)
			print(f"Attempting to load and visualize sample index: {sample_idx}")
			result = None
			try:
				result = dataset[sample_idx]
			except Exception as e:
				print(f"Error loading index {sample_idx}: {e}")
				result = None

			# Check if result is valid (tuple of 3, none are None)
			if result is None or not (isinstance(result, tuple) and len(result) == 3 and all(item is not None for item in result)):
				print(f"Failed to load a valid sample (image, landmarks, label) from index {sample_idx}. Result: {result}")
			else:
				print(f"Successfully loaded sample index: {sample_idx}")
				image_tensor, landmarks_tensor, label_index_tensor = result

				print("Sample image tensor shape:", image_tensor.shape)
				print("Sample landmarks tensor shape:", landmarks_tensor.shape)
				print("Sample label index tensor:", label_index_tensor.item())

				selected_sample_info = dataset.samples[sample_idx]
				original_image_path = selected_sample_info['image_path']
				label_idx = label_index_tensor.item()
				gesture_name = dataset.index_to_label.get(label_idx, "Unknown")
				ann_id = selected_sample_info['ann_id']
				bbox_rel = selected_sample_info['bbox_rel']

				print(f"Annotation ID: {ann_id}, Gesture Label: {gesture_name} (Index: {label_idx})")

				original_image = cv2.imread(original_image_path)
				if original_image is not None:
					original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
					h_orig, w_orig, _ = original_image_rgb.shape

					cropped_image_np = image_tensor.permute(1, 2, 0).numpy()
					cropped_image_np = (cropped_image_np * 255).astype(np.uint8)
					h_crop, w_crop, _ = cropped_image_np.shape

					# Convert landmarks tensor back for display
					landmarks_plot = landmarks_tensor.numpy() * output_size

					# Recompute bbox_abs for plotting context
					x_rel, y_rel, bw_rel, bh_rel = bbox_rel
					x1_abs = x_rel * w_orig
					y1_abs = y_rel * h_orig
					x2_abs = (x_rel + bw_rel) * w_orig
					y2_abs = (y_rel + bh_rel) * h_orig
					bbox_abs = [x1_abs, y1_abs, x2_abs, y2_abs]

					# --- Plotting ---
					_, crop_coords_viz, scales_viz = crop_hand(
						original_image_rgb,
						bbox_abs,
						output_size=output_size,
						padding_factor=args.crop_padding_factor
					)

					bbox_crop_xyxy = None
					if crop_coords_viz and scales_viz:
						try:
							bbox_crop_xyxy = transform_bbox_to_crop(
								bbox_abs=bbox_abs,
								crop_coords=crop_coords_viz,
								scales=scales_viz,
								crop_size=(h_crop, w_crop)
							)
						except Exception as plot_e:
							print(f"Warning: Could not calculate bbox_crop_xyxy for plotting. {plot_e}")


					titles = {
						'title': f"Ann ID: {ann_id}, Index: {sample_idx}",
						'original': f"Original Image\n({w_orig}x{h_orig})",
						'cropped': f"Cropped Image\nGesture: {gesture_name} (Idx: {label_idx})\n({w_crop}x{h_crop}) Padding: {args.crop_padding_factor}"
					}

					plot_hand_data(
						original_image=original_image_rgb,
						cropped_image=cropped_image_np,
						landmarks_crop=landmarks_plot,
						bbox_original=bbox_abs,
						bbox_crop=bbox_crop_xyxy,
						titles=titles,
						landmark_size=80,
						show_labels=args.show_labels
					)
				else:
					print(f"Error: Could not load original image at {original_image_path}")

		else:
			print(f"Dataset for split '{split}' is empty. Cannot retrieve a sample.")
			print(f"Please check JSON files and data structure in: {os.path.abspath(os.path.join(dataset_root, 'annotations', split))}")
			if dataset.num_classes == 0:
				print("Check if any gesture names were found during label map creation.")

	except FileNotFoundError:
		print(f"Error: Root directory or annotations directory not found. Searched in {os.path.abspath(dataset_root)}")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")