import os
import cv2
import time
import torch
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec

from dataset import crop_hand, HagridDataset
from torch_utils import get_device, optimize_for_device
from landmark import MediapipeHandsLandmarks, MODEL_CONFIG
from utils import plot_hand_data, transform_bbox_to_crop

HAND_CONNECTIONS = [
	(0, 1), (1, 2), (2, 3), (3, 4), # Thumb
	(0, 5), (5, 6), (6, 7), (7, 8), # Index finger
	(9, 10), (10, 11), (11, 12), # Middle finger
	(13, 14), (14, 15), (15, 16), # Ring finger
	(0, 17), (17, 18), (18, 19), (19, 20), # Pinky
	(0, 9), (9, 13), (13, 17), # Palm
	(5, 9), (9, 13), (13, 17) # Palm
]

def draw_landmarks_on_image(
	image,
	landmarks_pred,
	landmarks_gt=None,
	show_labels=False,
	draw_connections=False,
	connections=None,
	pred_color=(0, 0, 255),
	gt_color=(0, 255, 0),
	connection_color=(255, 255, 255),
	size=2,
	connection_thickness=3
):
	"""Draws predicted landmarks and optionally ground truth landmarks and connections on an image."""
	img_h, img_w = image.shape[:2]

	# Draw ground truth landmarks
	if landmarks_gt is not None:
		for i, (x, y) in enumerate(landmarks_gt):
			if 0 <= x < img_w and 0 <= y < img_h:
				cv2.circle(image, (int(x), int(y)), size + 1, gt_color, -1)

	# Draw Predicted landmarks
	valid_landmarks = []
	for i, (x, y) in enumerate(landmarks_pred):
		if 0 <= x < img_w and 0 <= y < img_h:
			landmark_point = (int(x), int(y))
			valid_landmarks.append(landmark_point)
			cv2.circle(image, landmark_point, size, pred_color, -1)
			if show_labels:
				cv2.putText(
					image, str(i), (int(x) + size, int(y) + size),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
				)
		else:
			valid_landmarks.append(None)

	# Draw connections if enabled
	if draw_connections and connections and len(valid_landmarks) > 0:
		for connection in connections:
			start_idx, end_idx = connection
			if start_idx < len(valid_landmarks) and end_idx < len(valid_landmarks):
				start_point = valid_landmarks[start_idx]
				end_point = valid_landmarks[end_idx]
				if start_point and end_point:
					cv2.line(image, start_point, end_point, connection_color, connection_thickness)

	return image

def run_webcam_inference(
	model, 
	device, 
	output_size, 
	show_labels
):
	"""Run inference on live webcam feed
		- Mediapipe hands for hand detection
		- Trained landmark estimation model for hand pose estimation.
	"""
	mp_hands = mp.solutions.hands
	hands = mp_hands.Hands(
		static_image_mode=False,
		max_num_hands=2, # Allow multiple hands
		min_detection_confidence=0.7,
		min_tracking_confidence=0.5
	)

	# --- Webcam  ---
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Error: Could not open webcam.")
		return

	prev_time = 0
	print("Starting webcam feed... Press 'q' to quit.")

	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			continue

		frame = cv2.flip(frame, 1)
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_height, frame_width, _ = frame.shape
		results = hands.process(frame_rgb)

		# Calculate FPS
		curr_time = time.time()
		fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
		prev_time = curr_time
		cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		# --- Hand Detection and Landmark Prediction ---
		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				# Get Bounding Box from MediaPipe Landmarks
				x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
				y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]

				if not x_coords or not y_coords: 
					continue

				x_min, x_max = min(x_coords), max(x_coords)
				y_min, y_max = min(y_coords), max(y_coords)

				padding = 30
				x_min = max(0, x_min - padding)
				y_min = max(0, y_min - padding)
				x_max = min(frame_width, x_max + padding)
				y_max = min(frame_height, y_max + padding)

				bbox_abs = [int(x_min), int(y_min), int(x_max), int(y_max)]

				# Crop Hand
				crop_img, crop_coords, scales = crop_hand(frame_rgb, bbox_abs, output_size=output_size)
				if crop_img is None: 
					continue
				crop_img_tensor = torch.from_numpy(crop_img).permute(2, 0, 1).float() / 255.0
				crop_img_tensor = crop_img_tensor.unsqueeze(0).to(device)

				# Run inference using trained model
				with torch.no_grad():
					predicted_landmarks_norm = model(crop_img_tensor)
				predicted_landmarks_norm = predicted_landmarks_norm.squeeze(0).cpu().numpy()

				# Interpolate predicted landmarks back to original frame coordinates
				offset_x, offset_y, _, _ = crop_coords
				scale_x, scale_y = scales
				landmarks_orig_frame = []
				for point in predicted_landmarks_norm:
					crop_x = point[0] * output_size
					crop_y = point[1] * output_size
					orig_x = (crop_x / scale_x) + offset_x
					orig_y = (crop_y / scale_y) + offset_y
					landmarks_orig_frame.append([orig_x, orig_y])

				frame = draw_landmarks_on_image(
					frame, 
					landmarks_orig_frame, 
					show_labels=show_labels, 
					size=13,
					draw_connections=True,
					connections=HAND_CONNECTIONS,
					connection_color=(0, 255, 0)
				)

		cv2.imshow('MediaPipe Hands + Custom Landmarks', frame)

		if cv2.waitKey(5) & 0xFF == ord('q'):
			break

	hands.close()
	cap.release()
	cv2.destroyAllWindows()
	print("Webcam feed stopped.")

def run_dataset_inference(
	model, 
	device, 
	output_size, 
	args
):
	"""Run inference on a image from the Hagriddataset."""
	print(f"Loading dataset from: {args.data_dir}, split: {args.split}")
	try:
		dataset = HagridDataset(
			root=args.data_dir,
			split=args.split,
			output_size=output_size,
			augment=False
		)
		if not dataset:
			print("Error: Dataset could not be loaded.")
			return
		if args.index < 0 or args.index >= len(dataset):
			print(f"Error: Index {args.index} is out of bounds for dataset split '{args.split}' (size: {len(dataset)}).")
			return

		print(f"Running inference on sample index: {args.index}")
		result = dataset[args.index]
		if result is None or any(item is None for item in result):
			print(f"Error: Failed to load sample at index {args.index}. It might be problematic or skipped.")
			return
			
		crop_img_tensor, landmarks_gt_norm, _, crop_coords, scales = result
		vis_crop_tensor = crop_img_tensor.clone()
		crop_img_tensor = crop_img_tensor.unsqueeze(0).to(device)

		with torch.no_grad():
			predicted_landmarks_norm = model(crop_img_tensor) # Output shape [1, 21, 2], normalized [0,1]
		predicted_landmarks_norm = predicted_landmarks_norm.squeeze(0).cpu().numpy()

		# Convert landmarks to absolute coordinates within the 256x256 crop
		landmarks_pred_abs_crop = (predicted_landmarks_norm * output_size).astype(np.int32)
		landmarks_gt_abs_crop = (landmarks_gt_norm.numpy() * output_size).astype(np.int32)

		# Prepare the 256x256 crop for display
		cropped_display_img = vis_crop_tensor.permute(1, 2, 0).cpu().numpy()
		cropped_display_img = (cropped_display_img * 255).astype(np.uint8)
		cropped_display_img = cv2.cvtColor(cropped_display_img, cv2.COLOR_RGB2BGR)

		# Draw the predicted and ground truth landmarks on the cropped image
		output_image = draw_landmarks_on_image(
			cropped_display_img.copy(),
			landmarks_pred=landmarks_pred_abs_crop,
			landmarks_gt=landmarks_gt_abs_crop,
			show_labels=args.labels,
			draw_connections=False
		)

		# Display the cropped image with landmarks
		cv2.imshow(f'Image input with Landmarks (Split: {args.split}, Index: {args.index})', output_image)
		print(f"Displaying inference result for sample {args.index} (cropped view). Press any key to close window.")
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	except FileNotFoundError:
		print(f"Error: Dataset directory or annotation file not found at {args.data_dir}")
	except Exception as e:
		print(f"An error occurred during dataset inference: {e}")

def inference(args):
	"""Run inference on a trained model."""
	device = get_device()
	optimize_for_device(device)
	print(f"Using device: {device}")
	output_size = 256

	# --- Load Trained Model ---
	print(f"Loading model from checkpoint: {args.weights}")
	model = MediapipeHandsLandmarks(config=MODEL_CONFIG).to(device)
	try:
		checkpoint = torch.load(args.weights, map_location=device)
		if 'model_state_dict' in checkpoint:
			model.load_state_dict(checkpoint['model_state_dict'])
		model.eval()
		print("Model loaded successfully.")
	except FileNotFoundError:
		print(f"Error: Checkpoint file not found at {args.weights}")
		return
	except Exception as e:
		print(f"Error loading checkpoint: {e}")
		return

	# --- Mode Selection ---
	if args.webcam:
		run_webcam_inference(model, device, output_size, args.labels)
	else:
		run_dataset_inference(model, device, output_size, args)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run Hand Landmark Inference")
	parser.add_argument(
		"--weights",
		type=str,
		required=True,
		help="Path to the trained model checkpoint (.pth file)"
	)

	mode_group = parser.add_mutually_exclusive_group(required=False)
	mode_group.add_argument(
		"--webcam",
		action="store_true",
		help="Run inference on live webcam feed."
	)
	parser.add_argument(
		"--data_dir",
		type=str,
		default="data/hagrid_small",
		help="Root directory of the Hagrid dataset (required if --dataset is used)."
	)
	parser.add_argument(
		"--split",
		type=str,
		choices=["train", "val", "test"],
		help="Dataset split to use (required if --dataset is used)."
	)
	parser.add_argument(
		"--index",
		type=int,
		help="Index of the sample within the split to use (required if --dataset is used)."
	)

	parser.add_argument(
		"--labels",
		action="store_true",
		help="Show landmark index labels on the output visualization."
	)

	args = parser.parse_args()

	if not args.webcam:
		if args.split is None or args.index is None:
			parser.error("Dataset mode (default) requires --split and --index arguments.")

	inference(args)