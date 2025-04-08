# Posture and Gesture Recognition

This repository contains contains code for training and inference for the following:
- Blazepose for pose estimation
- Mediapipe for pose estimation and gesture recognition

## Installation

### Prerequistes

- git: [Link](https://git-scm.com/downloads) 
- Python 3.12 [Link](https://www.python.org/downloads/release/python-31210/)

### Setup

1. You will need to install Git and python 3.12 before proceeding



2. Clone the repository using git
	```bash
	git clone git@github.com:puravparab/Posture-Gesture-Recognition.git
	```

3. Change directory to Posture-Gesture-Recognition
	```
	cd Posture-Gesture-Recognition
	```

3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) for python package management

- Windows
	```bash
	pip install uv
	```
- Mac/Linux
	```bash
	curl -LsSf https://astral.sh/uv/install.sh | sh
	```
	
5. Download python dependencies
	```bash
	uv sync
	```
	or if that does not work
	```bash
	python3 -m uv sync
	```

## Inference

### Mediapipe

1. Pose estimation
	```bash
	uv run mediapipe/pose.py --threshold 0.1
	```
	Alternatively
	```bash
	python3 -m uv run mediapipe/pose.py --threshold 0.1
	```

2. Gesture Recognition
	```bash
	uv run mediapipe/gesture.py --threshold 0.1
	```
	Alternatively
	```bash
	python3 -m uv run mediapipe/gesture.py --threshold 0.1
	```
	
### Blazepose
1. Run this command to start the webcam
	```bash
	uv run blazepose/inference.py --weights weights/blazepose.pt --webcam --threshold 0
	```

2. Run this command if using a single image
	```bash
	uv run blazepose/inference.py --weights weights/blazepose.pt --input <path/to/input/image> --threshold 0
	```

	The output images with keypoints should be present in [output](./output/) directory


## Training

### Download the dataset

1. [Coco Keypoints 2017](https://cocodataset.org/#keypoints-2017)

 - Install unzip
	```bash
	sudo apt update
	sudo apt install unzip
	```

 - Download the train/val images and annotations

	```bash
	uv run data/coco_keypoints.py
	```

### BlazePose

1. Train model

	Log into weights and biases (if using --wandb)
	```bash
	wandb login
	```

	Start training script
	```bash
	uv run papers/blaze-pose/train.py --name "blazepose" --coco_dir data/coco_keypoints --epochs 10 --batch_size 64 --lr 0.001 --num_workers 3 --augment --wandb
	```

	Run or see [utils](/papers/blaze-pose/utils.py) for info about argument flags
	```bash
	uv run papers/blaze-pose/train.py --help
	```