import ssl
import torch
import argparse
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from dataset import HagridDataset, collate_filter_none
from torch_utils import get_device, optimize_for_device

ssl._create_default_https_context = ssl._create_unverified_context

class GestureClassifier(nn.Module):
	""" 
	Hand Gesture Classification model using a pre-trained ResNet-18 backbone.

	Args:
		num_classes (int): The number of gesture classes to predict.
		pretrained (bool): Whether to load pre-trained ImageNet weights using recommended API.
		freeze_backbone (bool): Whether to freeze the weights of the backbone initially.
	"""
	def __init__(
		self,
		num_classes: int,
		pretrained: bool = True,
		freeze_backbone: bool = True
	):
		super().__init__()
		self.num_classes = num_classes
		self.backbone_name = 'resnet18'

		# Load the pre-trained backbone
		print("Loading ResNet-18 backbone...")
		weights = models.ResNet18_Weights.DEFAULT if pretrained else None
		self.backbone = models.resnet18(weights=weights)
		num_features = self.backbone.fc.in_features

		# Replace the final fully connected layer
		self.backbone.fc = nn.Linear(num_features, num_classes)
		print(f"Replaced final layer for {num_classes} classes.")

		# Freeze backbone layers
		if freeze_backbone and pretrained:
			print(f"Freezing backbone layers of {self.backbone_name}.")
			for name, param in self.backbone.named_parameters():
				if not name.startswith('fc.'):
					param.requires_grad = False
			# Ensure the new fc layer is trainable (redundant but safe)
			for param in self.backbone.fc.parameters():
				param.requires_grad = True
			print("Classifier head ('fc') parameters set to trainable.")
		elif not freeze_backbone:
			print(f"Training entire {self.backbone_name} model (no freezing).")
		else: # Not pretrained, train all
			print(f"Training {self.backbone_name} model from scratch (no freezing).")


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through the backbone and classifier head.

		Args:
			x (torch.Tensor): Input batch of images, shape [B, 3, H, W].
		Returns:
			torch.Tensor: Raw logits for each class, shape [B, num_classes].
		"""
		return self.backbone(x)

	def unfreeze_backbone(self, stages_to_unfreeze: int = -1):
		"""
		Unfreezes stages of the ResNet backbone for fine-tuning.

		Args:
			stages_to_unfreeze (int): 
				Number of ResNet stages (layer blocks) to unfreeze from the end (1 to 4). 
				-1 means unfreeze all layers (including stem). 
				0 means only head is trainable.
		"""
		print(f"Unfreezing ResNet-18 backbone... Mode: {stages_to_unfreeze}")

		# Ensure the head is trainable
		for param in self.backbone.fc.parameters():
			param.requires_grad = True

		if stages_to_unfreeze == -1: # Unfreeze everything
			for param in self.backbone.parameters():
				param.requires_grad = True
			print("Unfroze all backbone layers (including stem).")
			return
		elif stages_to_unfreeze == 0: # Only head trainable
			print("Keeping only the head ('fc') trainable.")
			return

		layer_blocks = [self.backbone.layer4, self.backbone.layer3, self.backbone.layer2, self.backbone.layer1]
		stages_to_unfreeze = max(1, min(stages_to_unfreeze, len(layer_blocks)))
		layers_unfrozen_count = 0
		for i in range(stages_to_unfreeze):
			stage_index = len(layer_blocks) - 1 - i # 3, 2, 1, 0
			stage_name = f"layer{stage_index + 1}" # layer4, layer3, ...
			current_stage = layer_blocks[stage_index]

			all_frozen = all(not p.requires_grad for p in current_stage.parameters())

			for param in current_stage.parameters():
				param.requires_grad = True

			if all_frozen:
				print(f"Unfroze ResNet stage: {stage_name}")
				layers_unfrozen_count += 1

		print(f"Total ResNet stages unfrozen (excluding stem): {layers_unfrozen_count}")

	def count_parameters(self, trainable_only=True):
		"""Counts the total number of parameters in the model."""
		if trainable_only:
			return sum(p.numel() for p in self.parameters() if p.requires_grad)
		else:
			return sum(p.numel() for p in self.parameters())

if __name__ == '__main__':
	device = get_device()
	optimize_for_device(device)
	print(f"Using device: {device}")

	parser = argparse.ArgumentParser(description="Test GestureClassifier with HagridDataset.")
	parser.add_argument(
		"--dir",
		type=str,
		default="data/hagrid_small",
		help="Root directory of the dataset"
	)
	parser.add_argument(
		"--split",
		type=str,
		default="train",
		choices=["train", "val", "test"],
		help="Dataset split [train, val, test]"
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=4,
		help="Batch size for loading data"
	)
	args = parser.parse_args()

	print(f"Loading dataset info from: {args.dir}, split: {args.split}")
	try:
		dataset = HagridDataset(root=args.dir, split=args.split)
		num_gesture_classes = dataset.num_classes
		if num_gesture_classes == 0:
			raise ValueError("Dataset loaded but found 0 classes. Check annotations.")
		print(f"Determined number of classes: {num_gesture_classes}")
		print(f"Dataset size: {len(dataset)}")
	except (FileNotFoundError, ValueError) as e:
		print(f"Error loading dataset: {e}")
		print("Exiting.")
		exit()
	except Exception as e:
		print(f"An unexpected error occurred during dataset loading: {e}")
		print("Exiting.")
		exit()

	print(f"\nInstantiating Gesture Classifier with {num_gesture_classes} classes...")
	model = GestureClassifier(num_classes=num_gesture_classes, pretrained=True, freeze_backbone=True)
	model.to(device)
	print(f"Total Parameters: {model.count_parameters(trainable_only=False):,}")
	print(f"Trainable Parameters (Head Only): {model.count_parameters(trainable_only=True):,}")

	print(f"\nLoading data (batch size = {args.batch_size})...")
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=1,
		collate_fn=collate_filter_none
	)

	data_iter = iter(dataloader)
	batch = None
	try:
		while batch is None:
			batch = next(data_iter)
	except StopIteration:
		print("DataLoader exhausted or no valid batches found.")
		batch = None

	if batch:
		images, _, labels_gt = batch
		images = images.to(device)

		print("\n--- Forward Pass Test ---")
		print(f"Input shape: {images.shape}")

		model.eval()
		with torch.no_grad():
			output = model(images)

		print(f"Output shape: {output.shape}")
		if output.shape[0] > 0:
			print("Output logits:", output[0])
		else:
			print("Output tensor is empty.")
	else:
		print("Skipping forward pass test as no valid data batch was loaded.")