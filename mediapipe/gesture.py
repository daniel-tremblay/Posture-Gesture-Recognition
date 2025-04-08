import cv2
import time
import threading
import argparse
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_LANDMARK_NAMES = [
	'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip', 'index_finger_mcp',
	'index_finger_pip', 'index_finger_dip', 'index_finger_tip', 'middle_finger_mcp',
	'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip', 'ring_finger_mcp',
	'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip', 'pinky_mcp',
	'pinky_pip', 'pinky_dip', 'pinky_tip'
]
NUM_HAND_LANDMARKS = len(HAND_LANDMARK_NAMES)
MODEL_ASSET_PATH = 'mediapipe/gesture_recognizer.task'

latest_result = None
lock = threading.Lock()
# --- Result Callback ---
def save_result_callback(
	result: vision.GestureRecognizerResult,
	output_image: mp.Image,
	timestamp_ms: int
):
	"""Callback function to receive and store gesture recognition results."""
	global latest_result
	with lock:
		if result.gestures or result.hand_landmarks:
			latest_result = result
		else:
			latest_result = None

# --- Drawing Function ---
def draw_landmarks_styled(
	image,
	landmarks,
	landmark_color,
	connections,
	visibility_threshold=0.5
):
	"""Draws landmarks"""
	h, w, _ = image.shape
	if not landmarks:
		return

	# --- Draw Connections ---
	connection_color = (255, 255, 255)
	for connection in connections:
		start_idx = connection[0]
		end_idx = connection[1]
		if start_idx < len(landmarks) and end_idx < len(landmarks):
			start_lm = landmarks[start_idx]
			end_lm = landmarks[end_idx]
			start_pt = (int(start_lm.x * w), int(start_lm.y * h))
			end_pt = (int(end_lm.x * w), int(end_lm.y * h))
			cv2.line(image, start_pt, end_pt, connection_color, thickness=6, lineType=cv2.LINE_AA)


	# --- Draw Landmarks ---
	for idx, lm in enumerate(landmarks):
		center = (int(lm.x * w), int(lm.y * h))
		color = landmark_color
		cv2.circle(image, center, radius=8, color=color, thickness=-1, lineType=cv2.LINE_AA)
		cv2.circle(image, center, radius=10, color=color, thickness=2, lineType=cv2.LINE_AA)


# --- Webcam Processing ---
def process_webcam(
	num_hands: int = 2,
	min_detection_confidence: float = 0.5,
	min_tracking_confidence: float = 0.5,
	min_gesture_confidence: float = 0.5
):
	"""Processes webcam feed for gesture recognition."""
	global latest_result
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open webcam")

	# --- Create Gesture Recognizer ---
	try:
		base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
		options = vision.GestureRecognizerOptions(
			base_options=base_options,
			running_mode=vision.RunningMode.LIVE_STREAM,
			num_hands=num_hands,
			min_hand_detection_confidence=min_detection_confidence,
			min_hand_presence_confidence=min_tracking_confidence,
			min_tracking_confidence=min_tracking_confidence,
			result_callback=save_result_callback)
		recognizer = vision.GestureRecognizer.create_from_options(options)
		print("Gesture Recognizer created successfully.")
	except Exception as e:
		print(f"Error creating Gesture Recognizer: {e}")
		print("\nEnsure the model file 'gesture_recognizer.task' is accessible.")
		print("It might need to be downloaded by MediaPipe or placed manually.")
		return

	cv2.namedWindow('MediaPipe Gesture Recognition', cv2.WINDOW_NORMAL)
	frame_count = 0
	single_landmark_color = (0, 255, 0) # BGR format

	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			continue

		frame_count += 1
		# Flip the image horizontally
		frame = cv2.flip(frame, 1)
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

		# Get current time for async call
		timestamp_ms = int(time.time() * 1000)
		recognizer.recognize_async(mp_image, timestamp_ms)

		# --- Process and Draw Results ---
		display_frame = frame.copy()
		current_result = None
		with lock:
			if latest_result:
				current_result = latest_result

		# --- Create base frame and overlay --- 
		overlay = frame.copy()

		if current_result:
			h, w, _ = display_frame.shape
			alpha = 0.7

			# Draw "Press 'q' to quit"
			quit_text_x = 10
			quit_text_y = 25
			font_scale_quit = 0.8
			text_thickness_quit = 2
			text_color_quit = (255, 255, 255)
			cv2.putText(
				overlay, "Press 'q' to quit", (quit_text_x, quit_text_y),
				cv2.FONT_HERSHEY_SIMPLEX, font_scale_quit, text_color_quit, text_thickness_quit
			)

			# Iterate through detected hands
			for i in range(len(current_result.hand_landmarks)):
				landmarks = current_result.hand_landmarks[i]
				handedness = current_result.handedness[i][0]
				gestures = current_result.gestures[i]

				# --- Draw Landmarks & Connections ---
				draw_landmarks_styled(
					overlay,
					landmarks,
					single_landmark_color,
					mp.solutions.hands.HAND_CONNECTIONS
				)

				# --- Prepare Info Panel---
				panel_width = 300
				text_line_height = 30
				num_lines_display = 3 # Handedness, Gesture, Score
				panel_height = text_line_height * num_lines_display + 20

				# Determine panel position based on handedness
				is_left = handedness.category_name == 'Left'
				rect_x = w - panel_width - 10 if is_left else 10
				rect_y = quit_text_y + text_line_height

				# Draw panel
				cv2.rectangle(
					overlay, (rect_x, rect_y),
					(rect_x + panel_width, rect_y + panel_height),
					(70, 70, 70), -1
				)

				# Prepare text content
				y_offset = rect_y + 20
				text_x_start = rect_x + 15
				font_scale_info = 0.8
				text_thickness = 2
				text_color = (255, 255, 255)
				low_conf_color = (200, 200, 200)

				# Display Handedness - Flipping the name for display
				display_handedness = 'Right' if handedness.category_name == 'Left' else 'Left'
				hand_text = f"Hand: {display_handedness} ({handedness.score:.2f})"
				cv2.putText(
					overlay, hand_text, (text_x_start, y_offset),
					cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, text_color, text_thickness
				)
				y_offset += text_line_height

				# Display Top Gesture
				if gestures:
					top_gesture = gestures[0]
					gesture_text = f"Gesture: {top_gesture.category_name}"
					gesture_color = text_color if top_gesture.score >= min_gesture_confidence else low_conf_color
					cv2.putText(
						overlay, gesture_text, (text_x_start, y_offset),
						cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, gesture_color, text_thickness
					)
					y_offset += text_line_height

					score_text = f"Confidence: {top_gesture.score:.2f}"
					cv2.putText(
						overlay, score_text, (text_x_start, y_offset),
						cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, gesture_color, text_thickness
					)

				else:
					cv2.putText(
						overlay, "Gesture: None", (text_x_start, y_offset),
						cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, text_color, text_thickness
					)

			display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)

		else:
			# Display 'No hands detected' if result is None or empty
			cv2.putText(
				display_frame, "No hands detected", (10, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
			)

		# Display the frame
		cv2.imshow('MediaPipe Gesture Recognition', display_frame)

		# Break loop on 'q' press
		if cv2.waitKey(5) & 0xFF == ord('q'):
			break

	# Release resources
	recognizer.close()
	cap.release()
	cv2.destroyAllWindows()
	print("Cleaned up resources.")

# --- Main Function ---
def main():
	parser = argparse.ArgumentParser(description='MediaPipe Gesture Recognition Webcam Demo')
	parser.add_argument(
		'--num_hands', type=int, default=2,
		help='Maximum number of hands to detect.'
	)
	parser.add_argument(
		'--min_detection_confidence', type=float, default=0.5,
		help='Minimum confidence value for hand detection.'
	)
	parser.add_argument(
		'--min_tracking_confidence', type=float, default=0.5,
		help='Minimum confidence value for hand tracking/presence.'
	)
	parser.add_argument(
		'--threshold', type=float, default=0.5,
		help='Minimum confidence threshold for displaying recognized gestures.'
	)
	args = parser.parse_args()

	print("Starting webcam gesture recognition...")
	print("Press 'q' to quit.")

	process_webcam(
		num_hands=args.num_hands,
		min_detection_confidence=args.min_detection_confidence,
		min_tracking_confidence=args.min_tracking_confidence,
		min_gesture_confidence=args.threshold
	)

if __name__ == '__main__':
	main()
