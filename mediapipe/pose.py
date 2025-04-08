import cv2
import mediapipe as mp
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

POSE_LANDMARK_NAMES = [
	'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
	'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
	'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
	'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
	'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
	'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
	'right_heel', 'left_foot_index', 'right_foot_index'
]
NUM_POSE_LANDMARKS = len(POSE_LANDMARK_NAMES)


def process_webcam(
	model_complexity: int = 1,
	min_detection_confidence: float = 0.5,
	min_tracking_confidence: float = 0.5,
	visibility_threshold: float = 0.5
):
	"""
	Processes webcam feed for pose estimation using MediaPipe

	Args:
		model_complexity: Complexity of the pose landmark model (0, 1, or 2).
		min_detection_confidence: Min confidence for person detection ([0.0, 1.0]).
		min_tracking_confidence: Min confidence for landmark tracking ([0.0, 1.0]).
		visibility_threshold: Min visibility score for drawing landmarks ([0.0, 1.0]).
	"""
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open webcam")

	cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)
	
	landmark_colors = plt.cm.plasma(np.linspace(0, 1, NUM_POSE_LANDMARKS)) * 255

	with mp_pose.Pose(
		model_complexity=model_complexity,
		min_detection_confidence=min_detection_confidence,
		min_tracking_confidence=min_tracking_confidence
	) as pose:
		try:
			while cap.isOpened():
				success, frame = cap.read()
				if not success:
					print("Ignoring empty camera frame.")
					continue

				# Flip the image horizontally
				frame = cv2.flip(frame, 1)
				frame.flags.writeable = False
				image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

				# Process the image and detect pose
				results = pose.process(image_rgb)

				# Prepare frame for drawing
				frame.flags.writeable = True

				h, w, _ = frame.shape

				# Draw the pose annotation if landmarks are detected
				if results.pose_landmarks:
					landmarks = results.pose_landmarks.landmark

					# --- Draw Connections ---
					for i, j in mp_pose.POSE_CONNECTIONS:
						if i < NUM_POSE_LANDMARKS and j < NUM_POSE_LANDMARKS:
							lm1 = landmarks[i]
							lm2 = landmarks[j]
							# Draw line only if both landmarks are visible
							if lm1.visibility >= visibility_threshold and lm2.visibility >= visibility_threshold:
								pt1 = (int(lm1.x * w), int(lm1.y * h))
								pt2 = (int(lm2.x * w), int(lm2.y * h))
								color = tuple(map(int, landmark_colors[i][:3]))
								cv2.line(frame, pt1, pt2, color, thickness=6, lineType=cv2.LINE_AA)

					# --- Draw Landmarks ---
					for idx, lm in enumerate(landmarks):
						if lm.visibility >= visibility_threshold:
							center = (int(lm.x * w), int(lm.y * h))
							color = tuple(map(int, landmark_colors[idx][:3]))
							cv2.circle(frame, center, radius=8, color=color, thickness=-1, lineType=cv2.LINE_AA)

					# --- Create Keypoint Info Overlay ---
					info_panel_width = 350
					text_line_height = 25
					num_lines_display = NUM_POSE_LANDMARKS + 3
					info_panel_height = text_line_height * num_lines_display + 20
					overlay = frame.copy()
					alpha = 0.6 # Transparency factor
					rect_x, rect_y = 10, 10
					cv2.rectangle(
						overlay, (rect_x, rect_y),
						(rect_x + info_panel_width, rect_y + info_panel_height),
						(50, 50, 50),
						-1
					)

					frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

					# Draw keypoint information text
					y_offset = rect_y + 20
					text_x_indicator = rect_x + 15
					text_x_name = rect_x + 35
					font_scale_info = 0.6
					text_thickness = 2
					text_color = (255, 255, 255)

					cv2.putText(
						frame, "Press 'q' to quit", (text_x_indicator, y_offset),
						cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, text_color, text_thickness
					)
					y_offset += text_line_height

					for i, lm in enumerate(landmarks):
						name = POSE_LANDMARK_NAMES[i]
						color_bgr = tuple(map(int, landmark_colors[i][:3]))
						visibility = lm.visibility

						# Draw color indicator circle
						if visibility > 0.05:
							 cv2.circle(frame, (text_x_indicator, y_offset - 4), 7, color_bgr, -1)

						# Draw text (name: visibility)
						text = f"{name}: {visibility:.2f}"
						text_display_color = text_color if visibility >= visibility_threshold else (150, 150, 150)
						cv2.putText(
							frame, text, (text_x_name, y_offset),
							cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, text_display_color, text_thickness
						)
						y_offset += text_line_height
				else:
					 cv2.putText(frame, "No pose detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

				cv2.imshow('MediaPipe Pose', frame)

				# Break loop on 'q' press
				if cv2.waitKey(5) & 0xFF == ord('q'):
					break
		finally:
			cap.release()
			cv2.destroyAllWindows()

def main():
	parser = argparse.ArgumentParser(description='MediaPipe Pose Estimation Webcam Demo with Custom Viz')
	parser.add_argument(
		'--model_complexity', type=int, default=1, choices=[0, 1, 2],
		help='Set model complexity: 0 (fastest), 1 (default), 2 (most accurate).'
	)
	parser.add_argument(
		'--min_detection_confidence', type=float, default=0.5,
		help='Minimum confidence for person detection (0.0-1.0).'
	)
	parser.add_argument(
		'--threshold', type=float, default=0.5,
		help='Minimum visibility score for drawing landmarks/connections (0.0-1.0).'
	)
	args = parser.parse_args()

	print("Starting webcam pose estimation...")
	print("Press 'q' to quit.")

	process_webcam(
		model_complexity=args.model_complexity,
		min_detection_confidence=args.min_detection_confidence,
		min_tracking_confidence=args.threshold,
		visibility_threshold=args.threshold
	)

if __name__ == '__main__':
	main()
