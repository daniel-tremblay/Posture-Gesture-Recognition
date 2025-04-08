import os
import cv2
import torch
import argparse
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from model import BlazePose
from torch_utils import get_device

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

COCO_KEYPOINT_NAMES = [
	'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
	'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
	'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
	'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_CONNECTION_PAIRS = [
	(0, 1), (0, 2), (1, 3), (2, 4),  # Head
	(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Torso + Arms
	(5, 11), (6, 12), (11, 12),  # Torso + Hips
	(11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def load_model_from_checkpoint(
	checkpoint_path: str,
	device: torch.device
) -> tuple[BlazePose, dict]:
	"""Loads the BlazePose model and its config from a checkpoint file."""
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

	print(f"Loading checkpoint from: {checkpoint_path}")
	checkpoint = torch.load(checkpoint_path, map_location=device)

	if 'config' not in checkpoint:
		raise KeyError("Checkpoint does not contain model configuration ('config').")
	if 'state_dict' not in checkpoint:
		raise KeyError("Checkpoint does not contain model state dictionary ('state_dict').")

	config = checkpoint['config']
	model_config = config.get('model')
	if model_config is None:
		raise KeyError("Checkpoint config does not contain 'model' section.")

	required_keys = ['num_keypoints', 'activation', 'dropout_p', 'channels', 'input_size']
	for key in required_keys:
		if key not in model_config:
			raise KeyError(f"Model configuration in checkpoint is missing required key: '{key}'")

	model = BlazePose(config=model_config)
	model.load_state_dict(checkpoint['state_dict'])
	model.to(device)
	model.eval()
	print("Model loaded successfully.")
	return model, config


def detect_face_mediapipe(
	image_rgb: np.ndarray,
	min_detection_confidence: float = 0.5
) -> tuple[dict | None, np.ndarray]:
	"""
	Detects the primary face in an image using MediaPipe Face Detection.

	Args:
		image_rgb: Input image in RGB format (H, W, C).
		min_detection_confidence: Minimum confidence score for a detection.

	Returns:
		A dictionary containing the primary face detection details (bounding box,
		score) or None if no face is detected, and the image with detections drawn (optional).
	"""
	primary_detection = None
	highest_score = 0.0
	height, width, _ = image_rgb.shape
	# image_with_detections = image_rgb.copy() # Uncomment to draw detections

	with mp_face_detection.FaceDetection(
		model_selection=1, # 0 for short-range (2m), 1 for full-range (5m)
		min_detection_confidence=min_detection_confidence
	) as face_detection:
		# Process the image
		results = face_detection.process(image_rgb)

		# Extract detections
		if results.detections:
			for detection in results.detections:
				score = detection.score[0]
				if score > highest_score:
					highest_score = score
					# Bounding box is relative to image dims, convert to pixels
					box = detection.location_data.relative_bounding_box
					xmin = int(box.xmin * width)
					ymin = int(box.ymin * height)
					w = int(box.width * width)
					h = int(box.height * height)
					primary_detection = {
						'box_pixels': (xmin, ymin, w, h), # x, y, width, height
						'score': score
					}
	return primary_detection


def estimate_person_bbox(
	face_box: tuple[int, int, int, int], # (x, y, w, h) in pixels
	img_w: int,
	img_h: int
) -> tuple[int, int, int, int]:
	"""
	Estimates a bounding box for the person based on the face bounding box.
	Uses heuristics and clamps the box to image boundaries.

	Args:
		face_box: Tuple containing (x, y, w, h) of the detected face in pixels.
		img_w: Width of the original image.
		img_h: Height of the original image.

	Returns:
		Tuple containing the estimated and clamped person bounding box
		(crop_x, crop_y, crop_w, crop_h).
	"""
	face_x, face_y, face_w, face_h = face_box

	center_x = face_x + face_w / 2
	center_y = face_y + face_h / 2

	body_h = face_h * 7.0
	body_w = face_w * 2.5
	top_y = center_y - (body_h / 2) * (1/3.5)
	left_x = center_x - body_w / 2

	# Clamp coordinates and dimensions
	crop_x = max(0, int(left_x))
	crop_y = max(0, int(top_y))
	# Calculate bottom-right corner and clamp
	br_x = min(img_w, int(left_x + body_w))
	br_y = min(img_h, int(top_y + body_h))

	# Final width and height after clamping
	crop_w = br_x - crop_x
	crop_h = br_y - crop_y

	# Ensure minimum size
	crop_w = max(1, crop_w)
	crop_h = max(1, crop_h)

	return crop_x, crop_y, crop_w, crop_h

def preprocess_image_aspect_ratio(
	image: np.ndarray, # Input image (RGB HWC)
	input_size: int
) -> tuple[torch.Tensor, np.ndarray, float, tuple[int, int], tuple[int, int]]:
	"""
	Resizes image to fit input_size while preserving aspect ratio, pads to square,
	normalizes, and converts to a tensor.

	Args:
		image: Input image (numpy array, RGB, HWC format).
		input_size: Target square size for the model input (e.g., 256).

	Returns:
		A tuple containing:
		- img_tensor: Preprocessed image tensor (1, C, H, W) for the model.
		- img_padded_rgb: The padded image as a NumPy array (H, W, C) in RGB.
		- scale: The scaling factor applied to the largest dimension.
		- pad_offset: Tuple (pad_top, pad_left) indicating padding added.
		- original_shape_before_resize: Tuple (h, w) of the image *before* resizing
		  (this could be the crop dimensions or full image dimensions).
	"""
	original_h, original_w = image.shape[:2]

	# Determine new size maintaining aspect ratio
	if original_h > original_w:
		new_h = input_size
		scale = input_size / original_h
		new_w = int(original_w * scale)
	else:
		new_w = input_size
		scale = input_size / original_w
		new_h = int(original_h * scale)

	# Resize
	img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

	# Create square canvas and pad
	img_padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
	pad_top = (input_size - new_h) // 2
	pad_left = (input_size - new_w) // 2
	img_padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = img_resized

	# Normalize and convert to tensor
	img_tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).float() / 255.0
	img_tensor = img_tensor.unsqueeze(0)

	return img_tensor, img_padded, scale, (pad_top, pad_left), (original_h, original_w)

def postprocess_keypoints_transformed(
	keypoints_pred: torch.Tensor, # Shape: [1, num_kpts, 3] (scaled 0-input_size)
	scale: float,
	pad_offset: tuple[int, int],
	original_shape_before_resize: tuple[int, int],
	crop_origin: tuple[int, int] | None = None # (x, y) of crop top-left, or None
) -> np.ndarray:
	"""
	Transforms keypoints from model output space back to original image coordinates.
	Handles padding removal, rescaling, and optional crop offset addition.

	Args:
		keypoints_pred: Raw keypoint predictions from the model [1, K, 3].
		scale: The scaling factor used during preprocessing resize.
		pad_offset: (pad_top, pad_left) padding added during preprocessing.
		original_shape_before_resize: (h, w) of the image *before* resizing
										(crop or full image).
		crop_origin: Optional (x, y) tuple of the crop's top-left corner in the
					 original full image. If None, assumes full image was processed.

	Returns:
		keypoints_original: Keypoints scaled to the original full image coordinates [K, 3].
	"""
	keypoints_np = keypoints_pred.squeeze(0).cpu().numpy().copy() # Shape: [K, 3]
	pad_top, pad_left = pad_offset
	
	# 1. Subtract padding offset (coordinates relative to resized image/crop)
	keypoints_np[:, 0] -= pad_left
	keypoints_np[:, 1] -= pad_top

	# 2. Rescale to original size (before resize) (coordinates relative to original crop or full image)
	# Avoid division by zero if scale is somehow zero
	if scale > 1e-6:
		keypoints_np[:, 0] /= scale
		keypoints_np[:, 1] /= scale
	else:
		print("Warning: scale factor is very small or zero during postprocessing.")
		# Set coords to 0 or handle as error? Setting to 0 for now.
		keypoints_np[:, :2] = 0

	# 3. Add crop offset (coordinates relative to original full image)
	if crop_origin:
		crop_x, crop_y = crop_origin
		keypoints_np[:, 0] += crop_x
		keypoints_np[:, 1] += crop_y
	return keypoints_np # Shape: [num_kpts, 3] (scaled to original full image)

def draw_keypoints(
	image: np.ndarray, # Original image (RGB or BGR)
	keypoints: np.ndarray, # Rescaled keypoints [num_kpts, 3] (x, y, visibility)
	connection_pairs: list[tuple[int, int]],
	keypoint_names: list[str],
	visibility_threshold: float = 0.3,
	keypoint_radius: int = 8,
	line_thickness: int = 6
) -> np.ndarray:
	"""Draws keypoints and connections onto the image."""
	draw_img = image.copy()
	num_kpts = len(keypoint_names)
	keypoint_colors = plt.cm.plasma(np.linspace(0, 1, num_kpts)) * 255

	# Draw connections
	for i, j in connection_pairs:
		if i < num_kpts and j < num_kpts:
			kp1 = keypoints[i]
			kp2 = keypoints[j]
			# Draw line only if both keypoints are above the visibility threshold
			if kp1[2] >= visibility_threshold and kp2[2] >= visibility_threshold:
				pt1 = (int(kp1[0]), int(kp1[1]))
				pt2 = (int(kp2[0]), int(kp2[1]))
				# Use the color of the starting keypoint for the line
				color = tuple(map(int, keypoint_colors[i][:3]))
				cv2.line(draw_img, pt1, pt2, color, line_thickness, cv2.LINE_AA)

	# Draw keypoints (circles)
	for i in range(num_kpts):
		kp = keypoints[i]
		if kp[2] >= visibility_threshold: # Check visibility
			center = (int(kp[0]), int(kp[1]))
			color = tuple(map(int, keypoint_colors[i][:3]))
			cv2.circle(draw_img, center, keypoint_radius, color, -1, cv2.LINE_AA)
			cv2.circle(draw_img, center, keypoint_radius + 2, color, 2, cv2.LINE_AA)

	return draw_img

def process_webcam(
	model: BlazePose,
	config: dict,
	device: torch.device,
	threshold: float = 0.3,
	face_conf: float = 0.5,
	no_crop: bool = False
):
	"""
	Process webcam feed and display pose estimation results in real-time.
	
	Args:
		model: Loaded BlazePose model
		config: Model configuration
		device: Device to run inference on
		threshold: Visibility threshold for keypoints
		face_conf: Face detection confidence threshold
		no_crop: Whether to skip face detection and cropping
	"""
	input_size = config['model']['input_size']
	cap = cv2.VideoCapture(0)
	
	if not cap.isOpened():
		raise RuntimeError("Could not open webcam")
	
	# Create a window for the webcam feed
	cv2.namedWindow('BlazePose Webcam', cv2.WINDOW_NORMAL)
	
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
				
			# Convert BGR to RGB
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			original_h, original_w = frame_rgb.shape[:2]
			
			# Process the frame
			crop_origin = None
			img_to_process = frame_rgb
			
			if not no_crop:
				face_detection_result = detect_face_mediapipe(frame_rgb, face_conf)
				if face_detection_result:
					face_box_pixels = face_detection_result['box_pixels']
					crop_x, crop_y, crop_w, crop_h = estimate_person_bbox(
						face_box_pixels, original_w, original_h
					)
					if crop_w > 1 and crop_h > 1:
						img_to_process = frame_rgb[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
						crop_origin = (crop_x, crop_y)
			
			# Preprocess and run inference
			img_tensor, img_padded_rgb, scale, pad_offset, shape_before_resize = preprocess_image_aspect_ratio(
				img_to_process, input_size
			)
			img_tensor = img_tensor.to(device)
			
			with torch.no_grad():
				keypoints_pred = model(img_tensor, training=False)
			
			# Post-process keypoints
			keypoints_final = postprocess_keypoints_transformed(
				keypoints_pred,
				scale,
				pad_offset,
				shape_before_resize,
				crop_origin
			)
			
			# Draw keypoints on the frame
			output_frame = draw_keypoints(
				frame,
				keypoints_final,
				COCO_CONNECTION_PAIRS,
				COCO_KEYPOINT_NAMES,
				visibility_threshold=threshold,
				keypoint_radius=8,
				line_thickness=6
			)
			
			# --- Keypoint Info---
			num_kpts_coco = len(COCO_KEYPOINT_NAMES)
			landmark_colors = plt.cm.plasma(np.linspace(0, 1, num_kpts_coco)) * 255

			info_panel_width = 350
			text_line_height = 25
			num_lines_display = num_kpts_coco + 1 # Landmarks + quit message
			info_panel_height = text_line_height * num_lines_display + 40
			overlay = output_frame.copy()
			alpha = 0.6 # Transparency factor
			rect_x, rect_y = 10, 10
			cv2.rectangle(
				overlay, (rect_x, rect_y),
				(rect_x + info_panel_width, rect_y + info_panel_height),
				(50, 50, 50),
				-1
			)

			output_frame = cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0)

			# Draw keypoint information text and indicators
			y_offset = rect_y + 20
			text_x_indicator = rect_x + 15
			text_x_name = rect_x + 35
			font_scale_info = 0.6
			text_thickness = 2
			text_color = (255, 255, 255)

			cv2.putText(
				output_frame, "Press 'q' to quit", (text_x_indicator, y_offset),
				cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, text_color, text_thickness
			)
			y_offset += text_line_height * 2

			# Draw landmark info
			for i, (name, kp) in enumerate(zip(COCO_KEYPOINT_NAMES, keypoints_final)):
				color_bgr = tuple(map(int, landmark_colors[i][:3]))
				visibility = kp[2]
				cv2.circle(output_frame, (text_x_indicator, y_offset - 4), 7, color_bgr, -1)
					
				text = f"{name}: {visibility:.2f}"
				text_display_color = text_color if visibility >= threshold else (150, 150, 150)
				cv2.putText(
					output_frame, text, (text_x_name, y_offset),
					cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, text_display_color, text_thickness
				)

				y_offset += text_line_height

			# Display frames
			cv2.imshow('BlazePose Webcam', output_frame)

			# Break loop on 'q' press
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
				
	finally:
		cap.release()
		cv2.destroyAllWindows()

def main():
	parser = argparse.ArgumentParser(description='BlazePose Inference Script with Face Detection Crop')
	parser.add_argument('--weights', type=str, required=True, help='Path to the BlazePose model checkpoint (.pt) file.')
	parser.add_argument('--input', type=str, help='Path to the input image file.')
	parser.add_argument('--webcam', action='store_true', help='Use webcam instead of input image.')
	parser.add_argument('--threshold', type=float, default=0.3, help='Visibility threshold for drawing keypoints/connections.')
	parser.add_argument('--face_conf', type=float, default=0.5, help='Minimum confidence for face detection.')
	parser.add_argument('--no_crop', action='store_true', help='Force processing the full image, skipping face detection.')
	parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output images. Defaults to "output/<input_basename>".')
	args = parser.parse_args()

	device = get_device()
	print(f"Using device: {device}")

	# Load model
	model, config = load_model_from_checkpoint(args.weights, device)
	input_size = config['model']['input_size']
	num_keypoints = config['model']['num_keypoints']
	if num_keypoints != 17:
		print(f"Warning: Model trained with {num_keypoints} keypoints, but COCO connections assume 17.")

	if args.webcam:
		process_webcam(
			model=model,
			config=config,
			device=device,
			threshold=args.threshold,
			face_conf=args.face_conf,
			no_crop=args.no_crop
		)
	else:
		if not args.input:
			raise ValueError("Either --input or --webcam must be specified")
			
		# --- 1. Initialization ---
		model, config = load_model_from_checkpoint(args.weights, device)
		input_size = config['model']['input_size']
		num_keypoints = config['model']['num_keypoints']
		if num_keypoints != 17:
			print(f"Warning: Model trained with {num_keypoints} keypoints, but COCO connections assume 17.")

		# --- 2. Image Loading ---
		if not os.path.exists(args.input):
			raise FileNotFoundError(f"Input image not found: {args.input}")
		img_bgr = cv2.imread(args.input)
		if img_bgr is None:
			raise ValueError(f"Failed to load image: {args.input}")
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		original_h, original_w = img_rgb.shape[:2]
		print(f"Loaded image: {args.input} ({original_w}x{original_h})")

		# --- Face Detection & Cropping Logic ---
		crop_origin = None
		img_to_process = img_rgb # Default to full image
		process_path = "full image" # For logging

		if not args.no_crop:
			print("Attempting face detection...")
			face_detection_result = detect_face_mediapipe(img_rgb, args.face_conf)

			if face_detection_result:
				face_box_pixels = face_detection_result['box_pixels']
				face_score = face_detection_result['score']
				print(f"Face detected with confidence {face_score:.2f} at {face_box_pixels}")

				# --- 4. Person BBox Estimation ---
				crop_x, crop_y, crop_w, crop_h = estimate_person_bbox(
					face_box_pixels, original_w, original_h
				)
				print(f"Estimated person crop box: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

				# --- 5. Image Cropping ---
				if crop_w > 1 and crop_h > 1: # Ensure valid crop
					img_to_process = img_rgb[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
					crop_origin = (crop_x, crop_y) # Store top-left corner
					process_path = f"crop ({crop_w}x{crop_h})"
				else:
					print("Warning: Estimated crop dimensions are invalid, falling back to full image.")
					process_path = "full image (crop failed)"
			else:
				print("No face detected, processing full image.")
				process_path = "full image (no face)"
		else:
			print("Skipping face detection (--no_crop specified).")
			process_path = "full image (skipped crop)"

		# --- 6. Input Tensor Preparation ---
		print(f"Preprocessing image via '{process_path}' path...")
		img_tensor, img_padded_rgb, scale, pad_offset, shape_before_resize = preprocess_image_aspect_ratio(
			img_to_process, input_size
		)
		img_tensor = img_tensor.to(device)
		print(f"  Original shape for processing: {shape_before_resize[1]}x{shape_before_resize[0]}")
		print(f"  Resized with scale: {scale:.4f}, Padding (top, left): {pad_offset}")
		print(f"  Final tensor shape: {img_tensor.shape}") # Should be [1, 3, input_size, input_size]

		# --- 7. BlazePose Model Inference ---
		print("Running BlazePose inference...")
		with torch.no_grad():
			keypoints_pred = model(img_tensor, training=False) # Shape: [1, K, 3]
		print(f"Inference complete. Raw output shape: {keypoints_pred.shape}")

		# --- Intermediate: Draw on Padded/Downscaled Input ---
		print("Drawing keypoints on downscaled/padded input...")
		keypoints_raw_np = keypoints_pred.squeeze(0).cpu().numpy() # Raw coords (0-input_size)
		output_image_downscaled = draw_keypoints(
			img_padded_rgb, # Draw on the padded RGB image
			keypoints_raw_np, # Use raw coordinates relative to input_size
			COCO_CONNECTION_PAIRS,
			COCO_KEYPOINT_NAMES,
			visibility_threshold=args.threshold
		)

		# --- 8. Keypoint Coordinate Transformation (Post-processing) ---
		print("Post-processing keypoints...")
		keypoints_final = postprocess_keypoints_transformed(
			keypoints_pred,
			scale,
			pad_offset,
			shape_before_resize,
			crop_origin
		)
		print(f"Transformed {len(keypoints_final)} keypoints to original image coordinates.")

		# Print the final keypoints
		print("\n--- Final Keypoint Results (Original Image Coords) ---")
		for i, kp in enumerate(keypoints_final):
			name = COCO_KEYPOINT_NAMES[i] if i < len(COCO_KEYPOINT_NAMES) else f"KP {i}"
			# Clamp coordinates for printing/sanity check
			kp_x = max(0, min(original_w - 1, kp[0]))
			kp_y = max(0, min(original_h - 1, kp[1]))
			print(f"{i:2d} {name:<15}: x={kp_x:.2f}, y={kp_y:.2f}, vis={kp[2]:.4f}")
		print("----------------------------------------------------\n")

		# --- 9. Visualization / Output ---
		print("Drawing final keypoints on original image...")
		output_image_viz = draw_keypoints(
			img_bgr,
			keypoints_final,
			COCO_CONNECTION_PAIRS,
			COCO_KEYPOINT_NAMES,
			visibility_threshold=args.threshold
		)

		# Define output path
		base, ext = os.path.splitext(os.path.basename(args.input))
		if args.output_dir is None:
			output_dir = os.path.join("output", base)
		else:
			output_dir = args.output_dir
		os.makedirs(output_dir, exist_ok=True)

		# Save the visualized image
		output_path_original = os.path.join(output_dir, f"{base}_pose_original{ext}")
		output_path_downscaled = os.path.join(output_dir, f"{base}_pose_downscaled{ext}")

		try:
			# Save original size visualization
			cv2.imwrite(output_path_original, output_image_viz)
			print(f"Output image with keypoints (original size) saved to: {output_path_original}")

			# Save downscaled/padded visualization
			output_image_downscaled_bgr = cv2.cvtColor(output_image_downscaled, cv2.COLOR_RGB2BGR)
			cv2.imwrite(output_path_downscaled, output_image_downscaled_bgr)
			print(f"Output image with keypoints (downscaled/padded) saved to: {output_path_downscaled}")
		except Exception as e:
			print(f"Error saving output image(s): {e}")

		print("Processing finished.")


if __name__ == '__main__':
	main() 