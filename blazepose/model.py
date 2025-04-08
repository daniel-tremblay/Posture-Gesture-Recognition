import time
import json
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple, List, Optional, Type
from types import SimpleNamespace
from torch_utils import get_device, optimize_for_device, get_activation

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
		super(ConvBlock, self).__init__()
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

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.activation(x)
		return x

class BlazePose(nn.Module):
	"""
	BlazePose model implementation based on the paper:
	"BlazePose: On-device Real-time Body Pose tracking"
	arXiv:2006.10204v1
	"""
	def __init__(self, config: Dict):
		super(BlazePose, self).__init__()
		self.num_keypoints: int = config['num_keypoints']
		self.activation: str = config['activation']
		self.dropout_p: float = config['dropout_p']
		self.channels: Dict[str, int] = config['channels']
		self.input_size: int = config['input_size']
		self.heatmap_offset_channels: int = 3 * self.num_keypoints # 66 for 17 keypoints
		self.device: torch.device = get_device()

		# === Encoder (Backbone) ===
		self.enc1 = ConvBlock(3, self.channels['enc1'], stride=2, activation=self.activation)
		self.enc2 = ConvBlock(self.channels['enc1'], self.channels['enc2'], stride=2, activation=self.activation)
		self.enc3 = ConvBlock(self.channels['enc2'], self.channels['enc3'], stride=2, activation=self.activation)
		self.enc4 = ConvBlock(self.channels['enc3'], self.channels['enc4'], stride=2, activation=self.activation)
		self.enc5 = ConvBlock(self.channels['enc4'], self.channels['enc5'], stride=2, activation=self.activation)


		# === Heatmap Branch (Left Column - Training Only) ===
		self.hm1 = ConvBlock(self.channels['enc5'], self.channels['hm1'], activation=self.activation)
		self.hm_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		# Input channels = hm1_out + enc4_out
		self.hm2 = ConvBlock(self.channels['hm1'] + self.channels['enc4'], self.channels['hm2'], activation=self.activation)
		self.hm_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)	
		# Input channels = hm2_out + enc3_out
		self.hm3 = ConvBlock(self.channels['hm2'] + self.channels['enc3'], self.channels['hm3'], activation=self.activation)
		self.hm_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		# Input channels = hm3_out + enc2_out
		self.hm4 = ConvBlock(self.channels['hm3'] + self.channels['enc2'], self.channels['hm4'], activation=self.activation)
		# Final convolution to produce heatmaps and offsets
		self.hm_final = nn.Conv2d(self.channels['hm4'], self.heatmap_offset_channels, kernel_size=1)

		# === Regression Branch (Right Column - Inference Path) ===
		self.reg1 = ConvBlock(self.channels['enc5'], self.channels['reg1'], activation=self.activation)
		self.reg_down1 = ConvBlock(self.channels['reg1'], self.channels['reg_down1'], stride=2, activation=self.activation)
		self.reg_down2 = ConvBlock(self.channels['reg_down1'], self.channels['reg_down2'], stride=2, activation=self.activation)
		self.reg_flatten = nn.Flatten()

		regressor_input_size: int = 2 * 2 * self.channels['reg_down2']
		regressor_hidden_size: int = self.channels['regressor_hidden']

		self.regressor = nn.Sequential(
			nn.Linear(regressor_input_size, regressor_hidden_size),
			get_activation(self.activation, inplace=True),
			nn.Dropout(p=self.dropout_p),
			nn.Linear(regressor_hidden_size, self.num_keypoints * 3)  # Ouput: (x, y, visibility)
		)

		# === Training-only Regression Path ===
		# Input channels = hm4_out + enc2_out
		self.train_reg1= ConvBlock(self.channels['hm4'] + self.channels['enc2'], self.channels['train_reg1'], activation=self.activation)
		# Input channels = train_reg1_pooled + enc3_out
		self.train_reg2 = ConvBlock(self.channels['train_reg1'] + self.channels['enc3'], self.channels['train_reg2'], activation=self.activation)
		# Input channels = train_reg2_pooled + enc4_out
		self.train_reg3 = ConvBlock(self.channels['train_reg2'] + self.channels['enc4'], self.channels['train_reg3'], activation=self.activation)
		# Input channels = train_reg3_pooled + enc5_out
		self.train_reg4 = ConvBlock(self.channels['train_reg3'] + self.channels['enc5'], self.channels['train_reg4'], activation=self.activation)
		
		# Output of train_reg4 (channels['train_reg4']) is then fed into the shared regression tail (reg_down1 onwards)
		# Ensure channels['train_reg4'] == channels['reg1'] for the path merge
		if self.channels['train_reg4'] != self.channels['reg1']:
			print(f"""
			Warning: Output channels of train_reg4 ({self.channels['train_reg4']}) 
			do not match input channels of reg_down1 ({self.channels['reg1']}). 
			This assumes reg_down1 starts from train_reg4 output during training.
			""")

		# Initialize weights
		self._initialize_weights()

	def _initialize_weights(self) -> None:
		"""Initialize weights for the model"""
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(
		self, x: torch.Tensor, 
		training: bool = True
	) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
		"""Forward pass through the model

		Args:
			x (torch.Tensor): Input image tensor (Batch, Channels, Height, Width).
			training (bool): Flag to indicate training mode (True) or inference mode (False).

		Returns:
			if training:
				heatmaps (torch.Tensor): Output heatmaps and offsets (Batch, num_kpts*3, H/4, W/4).
				keypoints (torch.Tensor): Output keypoints (Batch, num_kpts, 3).
			else (inference):
				keypoints (torch.Tensor): Output keypoints (Batch, num_kpts, 3).
		"""
		x = x.to(self.device)

		# === 1. Encoder (Backbone) ===
		x1 = self.enc1(x)
		x2 = self.enc2(x1)
		x3 = self.enc3(x2)
		x4 = self.enc4(x3)
		x5 = self.enc5(x4) # Final encoder features (8x8)


		if training:
			# === Heatmap Branch (Training Only) ===
			hm = self.hm1(x5)
			hm = self.hm_up1(hm)
			hm = torch.cat([hm, x4], dim=1) # Requires hm.channels[hm1] + channels[enc4]
			hm = self.hm2(hm)
			hm = self.hm_up2(hm)
			hm = torch.cat([hm, x3], dim=1) # Requires channels[hm2] + channels[enc3]
			hm = self.hm3(hm)
			hm = self.hm_up3(hm)
			hm = torch.cat([hm, x2], dim=1) # Requires channels[hm3] + channels[enc2]
			hm_features = self.hm4(hm) 			# Store features before final heatmap layer
			heatmaps = self.hm_final(hm_features)

			# Regression Branch (Training Path) ===
			# --- Calculate training-specific regression features ---
			reg_train = torch.cat([hm_features.detach(), x2.detach()], dim=1) # Requires channels[hm4] + channels[enc2]
			reg_train = self.train_reg1(reg_train)

			reg_train = F.max_pool2d(reg_train, 2)
			reg_train = torch.cat([reg_train, x3.detach()], dim=1) # Requires channels[train_reg1] + channels[enc3]
			reg_train = self.train_reg2(reg_train)

			reg_train = F.max_pool2d(reg_train, 2)
			reg_train = torch.cat([reg_train, x4.detach()], dim=1) # Requires channels[train_reg2] + channels[enc4]
			reg_train = self.train_reg3(reg_train)

			reg_train = F.max_pool2d(reg_train, 2)
			reg_train = torch.cat([reg_train, x5.detach()], dim=1) # Requires channels[train_reg3] + channels[enc5]
			reg_train_final = self.train_reg4(reg_train) # Output has channels['train_reg4']

			# --- Calculate inference path features (including reg1) ---
			reg_inf = self.reg1(x5.detach())
			combined_reg = reg_train_final + reg_inf

			# Feed combined features into the shared regression tail
			reg = self.reg_down1(combined_reg) # Use combined features
			reg = self.reg_down2(reg)
			reg = self.reg_flatten(reg)
			keypoints = self.regressor(reg)
			keypoints = keypoints.view(-1, self.num_keypoints, 3)

			# Scale coordinates to input image size and apply sigmoid to visibility
			coords_xy = keypoints[..., :2] * self.input_size
			coords_vis = keypoints[..., 2:]
			keypoints = torch.cat([coords_xy, coords_vis], dim=-1)

			return heatmaps, keypoints

		# === Inference Mode ===
		else:
			# Only the Regression Branch is used during inference
			reg = self.reg1(x5)
			reg = self.reg_down1(reg)
			reg = self.reg_down2(reg)
			reg = self.reg_flatten(reg)
			keypoints = self.regressor(reg)
			keypoints = keypoints.view(-1, self.num_keypoints, 3)

			# Scale coordinates to input image size and apply sigmoid to visibility
			coords_xy = keypoints[..., :2] * self.input_size
			coords_vis = keypoints[..., 2:]
			keypoints = torch.cat([coords_xy, coords_vis], dim=-1)

			return keypoints

	def count_parameters(
		self, 
		training: bool = False, 
		active_only: bool = False
	) -> int:
		"""
		Counts the total or active parameters in the model.

		Args:
			training (bool): 
				If True, counts parameters active during training.
				If False, counts parameters active during inference.
			active_only (bool): 
				If True, only counts parameters for the specified mode.
				If False, counts all parameters regardless of mode.

		Returns:
			int: The number of parameters.
		"""
		if not active_only:
			return sum(p.numel() for p in self.parameters() if p.requires_grad)

		active_params = 0
		# Parameters always active (Encoder)
		core_layers: List[nn.Module] = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
		active_params += sum(sum(p.numel() for p in layer.parameters() if p.requires_grad) for layer in core_layers)

		# Parameters active during INFERENCE (and also Training)
		# Note: This is the lightweight regression path starting from enc5
		inference_layers: List[nn.Module] = [self.reg1, self.reg_down1, self.reg_down2, self.regressor]
		active_params += sum(sum(p.numel() for p in layer.parameters() if p.requires_grad) for layer in inference_layers)

		if training:
			# Parameters active ONLY during TRAINING
			training_only_layers: List[nn.Module] = [
				self.hm1, self.hm2, self.hm3, self.hm4, self.hm_final, # Heatmap branch (Upsample has no params)
				self.train_reg1, self.train_reg2, self.train_reg3, self.train_reg4 # Training regression path
			]

			for layer in training_only_layers:
				if isinstance(layer, nn.Module) and list(layer.parameters(recurse=False)):
					active_params += sum(p.numel() for p in layer.parameters() if p.requires_grad)
				elif hasattr(layer, 'modules'):
					for sub_layer in layer.modules():
						if sub_layer is not layer and list(sub_layer.parameters(recurse=False)):
							# Avoid double counting if layer itself was already added
							if layer not in core_layers and layer not in inference_layers:
								active_params += sum(p.numel() for p in sub_layer.parameters() if p.requires_grad)

		# During inference, only core + inference layers are active
		else:
			# Recalculate for clarity, ensuring no training_only layers are included
			active_params = 0
			core_layers: List[nn.Module] = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
			active_params += sum(sum(p.numel() for p in layer.parameters() if p.requires_grad) for layer in core_layers)
			inference_layers: List[nn.Module] = [self.reg1, self.reg_down1, self.reg_down2, self.regressor]
			active_params += sum(sum(p.numel() for p in layer.parameters() if p.requires_grad) for layer in inference_layers)

		return active_params

def default_blazepose_config() -> Dict:
	"""Creates a default configuration dictionary for the BlazePose model."""
	config = {
		'input_size': 256,
		'batch_size': 32,
		'num_keypoints': 17,
		'activation': 'ReLU6',
		'dropout_p': 0.3,
		'channels': {
			'input_channels': 3,
			
			# Encoder
			'enc1': 16, # 256x256x3 -> 128x128x16
			'enc2': 32, # 128x128x16 -> 64x64x32
			'enc3': 64, # 64x64x32 -> 32x32x64
			'enc4': 128, # 32x32x64 -> 16x16x128
			'enc5': 192, # 16x16x128 -> 8x8x192

			# Heatmap Branch
			'hm1': 32, # 8x8x192 -> 8x8x32
			'hm2': 32, # 16x16x32 -> 16x16x32
			'hm3': 32, # 32x32x32 -> 32x32x32
			'hm4': 32, # 64x64x32 -> 64x64x32

			# Training-only Regression Path
			'train_reg1': 32,		# (Input: hm4 + enc2 = 32+32=64), 64x64x32 -> 64x64x32
			'train_reg2': 64,		# (Input: train_reg1_pooled + enc3 = 32+64=96) 32x32x96 -> 32x32x64
			'train_reg3': 128,	# (Input: train_reg2_pooled + enc4 = 64+128=192) 16x16x192 -> 16x16x128
			'train_reg4': 192,	# (Input: train_reg3_pooled + enc5 = 128+192=320) 8x8x320 -> 8x8x192
			
			# Regression Branch (Inference Path + Tail)
			'reg1': 192,	# Input from enc5 for inference path
			'reg_down1': 192, # 8x8x192 -> 4x4x192
			'reg_down2': 192, # 4x4x192 -> 2x2x192	
			'regressor_hidden': 256,
		}
	}

	# Check concat inputs match ConvBlock inputs
	assert config['channels']['hm1'] + config['channels']['enc4'] == 32 + 128, "HM2 input mismatch"
	assert config['channels']['hm2'] + config['channels']['enc3'] == 32 + 64, "HM3 input mismatch"
	assert config['channels']['hm3'] + config['channels']['enc2'] == 32 + 32, "HM4 input mismatch"

	assert config['channels']['hm4'] + config['channels']['enc2'] == 32 + 32, "TrainReg1 input mismatch"
	assert config['channels']['train_reg1'] + config['channels']['enc3'] == 32 + 64, "TrainReg2 input mismatch"
	assert config['channels']['train_reg2'] + config['channels']['enc4'] == 64 + 128, "TrainReg3 input mismatch"
	assert config['channels']['train_reg3'] + config['channels']['enc5'] == 128 + 192, "TrainReg4 input mismatch"

	# Check merge point for training regression path into main regression path tail
	assert config['channels']['train_reg4'] == config['channels']['reg1'], \
		f"Output of train_reg4 ({config['channels']['train_reg4']}) must match " \
		f"output of reg1 ({config['channels']['reg1']}) for training path merge."

	return config
	
if __name__ == '__main__':
	"""Example usage for testing"""

	config = default_blazepose_config()

	print("\n--- Using Configuration ---")
	print(json.dumps(config, indent=4))

	# Access values using attribute access now
	num_keypoints = config['num_keypoints']
	input_size = config['input_size']
	batch_size = config['batch_size']
	
	# Get device and optimize settings
	device = get_device()
	optimize_for_device(device)
	print(f"\nUsing device: {device}")

	# Create a random input tensor
	# Use config.channels['input_channels'] as channels is still a dict
	x = torch.randn(config['batch_size'], config['channels']['input_channels'], config['input_size'], config['input_size'], dtype=torch.float32, device=device)

	# --- Initialize model using the config ---
	model = BlazePose(config=config)
	model = model.to(device)

	# Print model architecture info
	print(f"\n--- Model Architecture Info ({model.num_keypoints} Keypoints) ---")
	total_params = model.count_parameters(active_only=False)
	# Correctly call count_parameters for training/inference active params
	train_active_params = model.count_parameters(training=True, active_only=True)
	infer_active_params = model.count_parameters(training=False, active_only=True)
	print(f"Total parameters: {total_params:,}")
	print(f"Active parameters during Training: {train_active_params:,}")
	print(f"Active parameters during Inference: {infer_active_params:,}")
	print(f"Parameter difference (Training - Inference): {train_active_params - infer_active_params:,}")
	print(f"Number of keypoints: {model.num_keypoints}")
	print(f"Input size: {model.input_size}x{model.input_size}")
	print(f"Heatmap/Offset output channels (Training): {model.heatmap_offset_channels}")
	print(f"Activation function: {model.activation}")
	print(f"Regressor dropout: {model.dropout_p}")

	try:
		# === Test Training Mode ===
		print("\n--- Testing Training Mode ---")
		model.train()
		with torch.no_grad():
			start_time = time.time()
			heatmaps, keypoints_train = model(x, training=True)
			training_time = time.time() - start_time
			print(f"Input shape: {x.shape}")
			print(f"Heatmaps shape: {heatmaps.shape}")
			print(f"Keypoints shape (Train): {keypoints_train.shape}")
			print(f"Training forward pass time: {training_time:.4f} seconds")

		# === Test Inference Mode ===
		print("\n--- Testing Inference Mode ---")
		model.eval()
		with torch.no_grad():
			start_time = time.time()
			keypoints_infer = model(x, training=False)
			inference_time = time.time() - start_time
			print(f"Input shape: {x.shape}")
			print(f"Keypoints shape (Infer): {keypoints_infer.shape}")
			print(f"Inference forward pass time: {inference_time:.4f} seconds")

	except Exception as e:
		print(f"\n--- Error during model execution ---")
		print(traceback.format_exc())