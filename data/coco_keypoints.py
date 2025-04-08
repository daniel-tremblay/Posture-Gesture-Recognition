# coco_keypoints.py
# This script downloads the COCO keypoints subset

import os
import time
import json
import shutil
import argparse
import asyncio
import aiohttp
from tqdm import tqdm
from pycocotools.coco import COCO
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

async def download_file(
	session: aiohttp.ClientSession, 
	url: str, 
	path: str, 
	retries: int = 3,
	desc: str = None
) -> bool:
	"""Download a file asynchronously with retries."""
	for attempt in range(retries):
		try:
			async with session.get(url) as response:
				if response.status == 200:
					total_size = int(response.headers.get('content-length', 0))
					with open(path, 'wb') as f:
						if desc:
							with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
								async for chunk in response.content.iter_chunked(8192):
									size = f.write(chunk)
									pbar.update(size)
						else:
							f.write(await response.read())
					return True
				await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
		except Exception as e:
			if attempt == retries - 1:
				print(f"Failed to download {url} after {retries} attempts: {e}")
			await asyncio.sleep(1 * (attempt + 1))
	return False

async def download_images(urls_and_paths: List[tuple], max_concurrent: int = 10):
	"""Download multiple images concurrently."""
	async with aiohttp.ClientSession() as session:
		tasks = []
		semaphore = asyncio.Semaphore(max_concurrent)
		
		async def bounded_download(url: str, path: str):
			async with semaphore:
				return await download_file(session, url, path)
		
		for url, path in urls_and_paths:
			if not os.path.exists(path):
				task = asyncio.create_task(bounded_download(url, path))
				tasks.append(task)
		
		if tasks:
			results = []
			for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading images"):
				results.append(await f)

async def download_coco_keypoints(
	output_dir='coco_keypoints', 
	download_train=True, 
	download_val=True
):
	"""
	Download COCO images that contain people with keypoint annotations.
	
	Args:
		output_dir: Directory to save the dataset
		download_train: Whether to download training set
		download_val: Whether to download validation set
	"""
	# Create output directories
	os.makedirs(f"{output_dir}/annotations", exist_ok=True)
	if download_train:
		os.makedirs(f"{output_dir}/train2017", exist_ok=True)
	if download_val:
		os.makedirs(f"{output_dir}/val2017", exist_ok=True)
	
	# Download annotation files
	print("Downloading annotation files...")
	annotations_zip = "annotations_trainval2017.zip"
	annotations_url = f"http://images.cocodataset.org/annotations/{annotations_zip}"
	
	# Download annotation files if they don't exist
	if not os.path.exists(annotations_zip):
		print(f"Downloading {annotations_url}...")
		async with aiohttp.ClientSession() as session:
			await download_file(session, annotations_url, annotations_zip, desc="Downloading annotations")
	
	# Extract annotations
	print("Extracting annotations...")
	os.system(f"unzip -q {annotations_zip} -d {output_dir}")
	
	# Move annotation files
	ann_files = []
	if download_train:
		ann_files.append('person_keypoints_train2017.json')
	if download_val:
		ann_files.append('person_keypoints_val2017.json')
	
	for file in ann_files:
		src = f"annotations/{file}"
		dst = f"{output_dir}/annotations/{file}"
		if os.path.exists(src) and not os.path.exists(dst):
			shutil.move(src, dst)

	# Process training set
	if download_train:
		print("Processing training set...")
		train_annot_file = f"{output_dir}/annotations/person_keypoints_train2017.json"
		train_coco = COCO(train_annot_file)
		
		# Get person category ID
		cat_ids = train_coco.getCatIds(catNms=['person'])
		
		# Get images with keypoint annotations
		img_ids = []
		ann_ids = train_coco.getAnnIds(catIds=cat_ids, iscrowd=False)
		anns = train_coco.loadAnns(ann_ids)
		
		# Filter out annotations without keypoints
		for ann in anns:
			if 'keypoints' in ann and sum(ann['keypoints'][2::3]) > 0:
				img_ids.append(ann['image_id'])
		
		# Remove duplicates
		img_ids = list(set(img_ids))
		print(f"Found {len(img_ids)} training images with keypoint annotations")
		
		# Download images
		urls_and_paths = []
		for img_id in img_ids:
			img_info = train_coco.loadImgs(img_id)[0]
			file_name = img_info['file_name']
			img_url = f"http://images.cocodataset.org/train2017/{file_name}"
			img_path = f"{output_dir}/train2017/{file_name}"
			urls_and_paths.append((img_url, img_path))
		await download_images(urls_and_paths, max_concurrent=20)
	
	# Process validation set
	if download_val:
		print("Processing validation set...")
		val_annot_file = f"{output_dir}/annotations/person_keypoints_val2017.json"
		val_coco = COCO(val_annot_file)
		
		# Get person category ID
		cat_ids = val_coco.getCatIds(catNms=['person'])
		
		# Get images with keypoint annotations
		img_ids = []
		ann_ids = val_coco.getAnnIds(catIds=cat_ids, iscrowd=False)
		anns = val_coco.loadAnns(ann_ids)
		
		# Filter out annotations without keypoints
		for ann in anns:
			if 'keypoints' in ann and sum(ann['keypoints'][2::3]) > 0:
				img_ids.append(ann['image_id'])
		
		# Remove duplicates
		img_ids = list(set(img_ids))
		print(f"Found {len(img_ids)} validation images with keypoint annotations")
		
		# Download images
		urls_and_paths = []
		for img_id in img_ids:
			img_info = val_coco.loadImgs(img_id)[0]
			file_name = img_info['file_name']
			img_url = f"http://images.cocodataset.org/val2017/{file_name}"
			img_path = f"{output_dir}/val2017/{file_name}"
			urls_and_paths.append((img_url, img_path))
		await download_images(urls_and_paths, max_concurrent=20)
	
	print(f"COCO keypoints dataset downloaded to {output_dir}")
	
	# Clean up
	if os.path.exists(annotations_zip):
		os.remove(annotations_zip)
	if os.path.exists("annotations"):
		shutil.rmtree("annotations")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Download COCO keypoints dataset')
	parser.add_argument('--output_dir', type=str, default='data/coco_keypoints', help='Output directory')
	parser.add_argument('--train', action='store_true', help='Download training set')
	parser.add_argument('--val', action='store_true', help='Download validation set')
	parser.add_argument('--all', action='store_true', help='Download both train and val sets')
	
	args = parser.parse_args()
	
	# If no specific set is selected, default to validation only
	if not (args.train or args.val or args.all):
		args.train = True
		args.val = True
	
	download_train = args.train or args.all
	download_val = args.val or args.all
	
	asyncio.run(download_coco_keypoints(args.output_dir, download_train, download_val))