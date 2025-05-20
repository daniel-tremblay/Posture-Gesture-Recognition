import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Any
from dataset import HagridDataset, collate_filter_none
from torch_utils import get_device, optimize_for_device

def collate_landmarks(
	batch: List[Optional[Tuple[Any, torch.Tensor, torch.Tensor]]]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
	"""
	Processes a batch of samples from the dataset. 
	- Keeps (landmarks, label) and filters out any sample where
	landmarks or label is None.

	Args:
		batch: 
			A list of tuples, where each tuple is expected to be (image_tensor, landmarks_tensor, label_tensor) or None. image_tensor is ignored.

	Returns:
		A tuple containing:
			- landmarks (torch.Tensor): Stacked landmark tensors [B, 21, 2]
			- labels (torch.Tensor): Stacked label tensors [B]
		Returns None if the batch is empty after filtering.
	"""
	valid_samples = []
	for sample in batch:
		if sample is not None and len(sample) == 3 and sample[1] is not None and sample[2] is not None:
			valid_samples.append((sample[1], sample[2]))
	if not valid_samples:
		return None
		
	try:
		landmarks, labels = zip(*valid_samples)
		return torch.stack(landmarks), torch.stack(labels)
	except Exception as e:
		print(f"Collate Error: Failed to stack batch components. Error: {e}")
		return None

class SkeletalGestureClassifier(nn.Module):
	"""
	Hand skeleton gesture classifier

	Takes 2D landmark coordinates (normalized to crop space [0,1]) as input. Converts landmarks to bone vectors

	Args:
		num_classes: Number of gesture categories to predict.
		use_bone_vectors: 
			If True, convert 21 joints to 20 bone vectors as input features.
			If False, use flattened joint coordinates directly.
		hidden_dims
		dropout_p
	"""
	_PARENTS: List[int] = [
		0, 1, 2, 3,	# Thumb: (0->1, 1->2, 2->3, 3->4)
		0, 5, 6, 7,	# Index finger: (0->5, 5->6, 6->7, 7->8)
		0, 9, 10, 11,	# Middle finger: (0->9, 9->10, 10->11, 11->12)
		0, 13, 14, 15,	# Ring finger: (0->13, 13->14, 14->15, 15->16)
		0, 17, 18, 19	# Pinky finger: (0->17, 17->18, 18->19, 19->20)
	]
	NUM_LANDMARKS = 21
	NUM_COORDINATES = 2

	def __init__(
		self,
		num_classes: int,
		hidden_dims: Optional[List[int]] = None,
		dropout_p: float = 0.2,
	):
		super().__init__()
		self.num_classes = num_classes
		# Combined input: 20 bone vectors * 2 coords + 21 landmarks * 2 coords = 40 + 42 = 82
		self.input_dim = ((self.NUM_LANDMARKS - 1) * self.NUM_COORDINATES ) + ( self.NUM_LANDMARKS * self.NUM_COORDINATES)

		# === MLP Head ===
		hidden_dims = hidden_dims if hidden_dims is not None else [128, 256, 128]
		layers: list[nn.Module] = []
		current_dim = self.input_dim
		for h_dim in hidden_dims:
			layers.append(nn.Linear(current_dim, h_dim, bias=False))
			layers.append(nn.BatchNorm1d(h_dim))
			layers.append(nn.ReLU(inplace=True))
			layers.append(nn.Dropout(dropout_p))
			current_dim = h_dim
		layers.append(nn.Linear(current_dim, num_classes))
		self.mlp = nn.Sequential(*layers)

		self._init_weights()

	def _init_weights(self):
		"""Initialize weights of the MLP head."""
		for m in self.mlp.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm1d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)

	def _to_bone_vectors(
		self, 
		joints: torch.Tensor
	) -> torch.Tensor:
		"""
		Converts absolute joint coordinates to bone vectors.
		A bone vector = child_joint_coordinates - parent_joint_coordinates

		Args:
			joints (torch.Tensor): Input tensor of joint coordinates [B, NUM_LANDMARKS, NUM_COORDINATES]

		Returns:
			torch.Tensor: Bone vectors flattened, [B, (NUM_LANDMARKS-1) * NUM_COORDINATES].
		"""
		parent_coords = joints[:, self._PARENTS, :] # Shape: [B, 20, 2]
		child_coords = joints[:, 1:self.NUM_LANDMARKS, :] # Shape: [B, 20, 2]
		bone_vectors = child_coords - parent_coords  # Shape: [B, 20, 2]
		return bone_vectors.flatten(start_dim=1) # Shape: [B, 40]

	def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through the classifier.

		Args:
			landmarks (torch.Tensor): 
				Input batch of landmark coordinates [B, NUM_LANDMARKS, NUM_COORDINATES] normalized to [0, 1] relative to the crop.

		Returns:
			torch.Tensor: Raw logits for each class, shape [B, num_classes].
		"""
		if landmarks.dim() != 3 or landmarks.shape[1] != self.NUM_LANDMARKS or landmarks.shape[2] != self.NUM_COORDINATES:
			raise ValueError(
				f"Expected input landmarks shape [B, {self.NUM_LANDMARKS}, {self.NUM_COORDINATES}], but got {landmarks.shape}"
			)

		bone_features = self._to_bone_vectors(landmarks) # Shape: [B, 40]

		# Convert absolute landmark coordinates to relative coordinates
		root_landmark_coords = landmarks[:, 0:1, :] # Shape: [B, 1, 2]
		relative_landmarks = landmarks - root_landmark_coords # Shape: [B, 21, 2]
		relative_coords = relative_landmarks.flatten(start_dim=1) # Shape: [B, 42]
		
		features = torch.cat((bone_features, relative_coords), dim=1) # Shape: [B, 82]
		
		logits = self.mlp(features)
		return logits

	def count_parameters(self, trainable_only: bool = True) -> int:
		"""Counts the total number of parameters in the model."""
		params = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
		return sum(p.numel() for p in params)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Test SkeletalGestureClassifier with HagridDataset.")
	parser.add_argument("--dir", type=str, default="data/hagrid_small", help="Root directory of the dataset")
	parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
	parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
	args = parser.parse_args()

	device = get_device()
	optimize_for_device(device)
	print(f"Using device: {device}")

	# Load dataset
	dataset = HagridDataset(root=args.dir, split=args.split)
	print(f"Loaded {len(dataset)} samples from {args.split} split.")

	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=1,
		collate_fn=collate_landmarks
	)

	# Instantiate model
	num_classes = dataset.num_classes
	model = SkeletalGestureClassifier(
		num_classes=num_classes,
		hidden_dims=[64, 128],
		dropout_p=0.2
	).to(device)
	print(f"Model instantiated with {num_classes} classes.")

	# Fetch a batch
	batch = None
	data_iter = iter(dataloader)
	try:
		while batch is None:
			batch = next(data_iter)
	except StopIteration:
		print("No valid batch found in the dataset.")
		batch = None

	if batch:
		landmarks, labels = batch
		landmarks = landmarks.to(device)
		labels = labels.to(device)
		print(f"Batch: landmarks shape {landmarks.shape}, labels shape {labels.shape}")
		model.eval()
		with torch.no_grad():
			logits = model(landmarks)
		print(f"Output logits shape: {logits.shape}")
	else:
		print("No valid data to test the model.")