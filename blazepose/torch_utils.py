import torch
import torch.nn as nn

def get_device() -> torch.device:
	"""Get the appropriate device (MPS, CUDA, or CPU) for PyTorch operations."""
	if torch.backends.mps.is_available():
		return torch.device("mps")
	elif torch.cuda.is_available():
		return torch.device("cuda")
	else:
		return torch.device("cpu")

def optimize_for_device(device: torch.device):
	"""Configure PyTorch settings for optimal performance on the given device."""
	if device.type == "mps":
		# Enable MPS-specific optimizations
		torch.backends.mps.enable_fallback_to_cpu = True
		torch.backends.mps.enable_async_execution = True
	elif device.type == "cuda":
		# Enable CUDA-specific optimizations
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.deterministic = False

def get_activation(name: str, inplace: bool = True) -> nn.Module:
	"""Gets an activation layer instance based on its name."""
	if name == 'ReLU':
		return nn.ReLU(inplace=inplace)
	elif name == 'ReLU6':
		return nn.ReLU6(inplace=inplace)
	else:
		raise ValueError(f"Unsupported activation function: {name}") 