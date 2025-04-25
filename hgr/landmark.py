import torch
import torch.nn as nn
from torch_utils import get_activation, get_device, optimize_for_device
from torch.utils.data import DataLoader

from dataset import HagridDataset, collate_filter_none

MODEL_CONFIG = {
	"activation": "ReLU",
	"dropout": 0.2,
	"num_landmarks": 21,
}

class ConvBlock(nn.Module):
	"""Convolutional block with BatchNorm and ReLU"""
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int = 3,
		stride: int = 1,
		activation: str = 'ReLU'
	):
		super().__init__()
		self.conv = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=kernel_size,
			stride=stride,
			padding=kernel_size // 2,
			bias=False
		)
		self.bn = nn.BatchNorm2d(out_channels)
		self.activation = get_activation(activation, inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.activation(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
	"""
	Basic residual block
	"""
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		stride: int = 1,
		activation: str = 'ReLU'
	):
		super().__init__()
		self.stride = stride
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.activation_fn = get_activation(activation, inplace=True)

		self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, activation=activation)
		self.conv2_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2_bn = nn.BatchNorm2d(out_channels)

		self.use_projection = (stride != 1) or (in_channels != out_channels)
		if self.use_projection:
			self.projection = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels)
			)
		else:
			self.projection = nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = x

		# Main path
		out = self.conv1(x)
		out = self.conv2_bn(self.conv2_conv(out))

		# Shortcut path
		identity = self.projection(identity)
		out += identity

		out = self.activation_fn(out)
		return out

class MediapipeHandsLandmarks(nn.Module):
	"""
	Mediapipe Hands Landmark model implementation based on the paper:
	"Mediapipe Hands: On-device Real-time Hand Tracking"
	arXiv:2006.10214
	"""
	def __init__(self, config: dict = MODEL_CONFIG):
		super().__init__()
		self.config = config
		num_landmarks = config["num_landmarks"]
		activation = config["activation"]
		dropout_p = config["dropout"]

		# === Feature Extractor===
		# Input: 256x256x3
		# Output: 128x128x32
		self.stem = ConvBlock(3, 32, kernel_size=3, stride=2, activation=activation)

		# Residual Stages
		self.stage1 = self._make_stage(32, 64, num_blocks=2, stride=2, activation=activation) # Output: 64x64x64
		self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2, activation=activation) # Output: 32x32x128
		self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2, activation=activation) # Output: 16x16x256
		self.stage4 = self._make_stage(256, 256, num_blocks=2, stride=1, activation=activation) # Output: 16x16x256

		# Final pooling
		self.avgpool = nn.AdaptiveAvgPool2d(1) # Output: 1x1x256

		# === Regression Head ===
		self.mlp = nn.Sequential(
			nn.Flatten(),
			nn.Linear(256, 512),
			get_activation(activation, inplace=True),
			nn.Dropout(p=dropout_p),
			nn.Linear(512, num_landmarks * 2) # (num_landmarks * 2) coordinates
		)

		self._init_weights()

	def _make_stage(
		self,
		in_channels: int,
		out_channels: int,
		num_blocks: int,
		stride: int,
		activation: str
	):
		"""Create a stage of residual blocks."""
		layers = []
		layers.append(ResidualBlock(in_channels, out_channels, stride=stride, activation=activation))
		for _ in range(1, num_blocks):
			layers.append(ResidualBlock(out_channels, out_channels, stride=1, activation=activation))
		return nn.Sequential(*layers)

	def _init_weights(self):
		"""Initialize weights of the network."""
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

		for m in self.modules():
			if isinstance(m, ResidualBlock):
				nn.init.constant_(m.conv2_bn.weight, 0) # Initialize gamma (weight) to 0

		if hasattr(self.mlp[-1], 'bias') and self.mlp[-1].bias is not None:
			nn.init.zeros_(self.mlp[-1].bias)

		if hasattr(self.mlp[-1], 'weight'):
			nn.init.zeros_(self.mlp[-1].weight)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem(x)
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		features = self.avgpool(x) # [B, C_out, 1, 1]
		landmarks = self.mlp(features) # [B, num_landmarks * 2]
		return landmarks.view(-1, self.config["num_landmarks"], 2) # Reshape to [Batch Size, Number of Landmarks, Coordinates (x, y)]

	def count_parameters(self) -> int:
		"""Counts the total number of trainable parameters in the model."""
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
	device = get_device()
	optimize_for_device(device)
	print(f"Using device: {device}")

	model = MediapipeHandsLandmarks()
	model.to(device)
	print(model)
	print("\nModel Config:", model.config)
	
	num_params = model.count_parameters()
	print(f"\nTotal Trainable Parameters: {num_params:,}")

	dataset_root = "data/hagrid_small"
	split = "train"
	batch_size = 4
	print(f"\nLoading {split} data from: {dataset_root}")

	try:
		dataset = HagridDataset(root=dataset_root, split=split)
		dataloader = DataLoader(
			dataset, batch_size=batch_size, shuffle=True, num_workers=0,
			collate_fn=collate_filter_none
		)
		print(f"Dataset size: {len(dataset)}")

		print(f"Fetching one batch (batch size = {batch_size})...")
		data_iter = iter(dataloader)

		batch = None
		try:
			while batch is None:
				batch = next(data_iter)
		except StopIteration:
			print("DataLoader exhausted or no valid batches found.")
			batch = None


		if batch:
			images, landmarks_gt = batch
			images = images.to(device)
			landmarks_gt = landmarks_gt.to(device)
			print(f"Input batch shape: {images.shape}")
			print(f"Ground Truth Landmarks shape: {landmarks_gt.shape}")

			print("\nRunning inference on the batch...")
			model.eval()
			with torch.no_grad():
				out = model(images)
			print(f"Output prediction shape: {out.shape}")
		else:
			print("Skipping inference as no valid data batch was loaded.")

	except FileNotFoundError:
		print(f"Error: Dataset directory not found at {dataset_root}")
	except Exception as e:
		print(f"An error occurred during testing: 	{e}")