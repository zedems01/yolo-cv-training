# Reorganize the 'valentynsichkar/traffic-signs-dataset-in-yolo-format' dataset into standard
# train/valid/test directory structure expected by YOLO training scripts.
# Usage:
# 1. As part of the 'download_dataset' utility in main.py:
#       - Call this function directly after downloading the dataset.
#       - The dataset will be restructured in-place and the path to the new root folder returned.
# 2. As a standalone script:
#       python reorganize_traffic_signs_dataset.py; from .../.../versions/4/
#       - After reorganization, use the restructured path with the '--local-dataset'
#         argument when launching model training.

import os
import shutil
import random
import logging
from typing import Dict, Any

def extract_filename(path):
    """Extract filename from full path"""
    return os.path.basename(path.strip())

def read_image_list(file_path):
    """Read image paths from a text file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def create_directory_structure(base_dir):
    """Create the directory structure"""
    dirs = [
        os.path.join(base_dir, 'train', 'images'),
        os.path.join(base_dir, 'train', 'labels'),
        os.path.join(base_dir, 'valid', 'images'),
        os.path.join(base_dir, 'valid', 'labels'),
        os.path.join(base_dir, 'test', 'images'),
        os.path.join(base_dir, 'test', 'labels'),
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def copy_file(src, dst):
    """Copy file from source to destination"""
    if os.path.exists(src):
        shutil.copy2(src, dst)
        return True
    return False

def reorganize_dataset(
    dataset_path: str,
    validation_split: float = 0.2,
    output_dir_name: str = 'traffic_signs',
) -> Dict[str, Any]:
    """
    Reorganize 'valentynsichkar/traffic-signs...' dataset into train/valid/test structure.
    
    Args:
        dataset_path: Root directory of the dataset (where train.txt, test.txt, and ts/ts/ are located)
        validation_split: Fraction of training data to use for validation (default: 0.2)
        output_dir_name: Name of the output directory (default: 'traffic_signs')
    Returns:
        Dictionary with reorganization results
    """
    random.seed(42)
    dataset_path = os.path.abspath(dataset_path)
    source_images_dir = os.path.join(dataset_path, 'ts', 'ts')
    train_txt = os.path.join(dataset_path, 'train.txt')
    test_txt = os.path.join(dataset_path, 'test.txt')
    output_dir = os.path.join(dataset_path, output_dir_name)
    
    logging.info("Starting dataset reorganization...")
    logging.info(f"Dataset path: {dataset_path}")
    logging.info(f"Source directory: {source_images_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    if not os.path.exists(train_txt):
        raise FileNotFoundError(f"train.txt not found at: {train_txt}")
    if not os.path.exists(test_txt):
        raise FileNotFoundError(f"test.txt not found at: {test_txt}")
    if not os.path.exists(source_images_dir):
        raise FileNotFoundError(f"Source images directory not found at: {source_images_dir}")
    
    logging.info("Creating directory structure...")
    create_directory_structure(output_dir)
    
    logging.info("Reading image lists...")
    train_images = read_image_list(train_txt)
    test_images = read_image_list(test_txt)
    
    logging.info(f"Training images: {len(train_images)}"; )
    logging.info(f"Test images: {len(test_images)}")
    
    random.shuffle(train_images)
    split_idx = int(len(train_images) * (1 - validation_split))
    train_split = train_images[:split_idx]
    valid_split = train_images[split_idx:]
    
    logging.info(f"Split results:")
    logging.info(f"  Train: {len(train_split)} images")
    logging.info(f"  Validation: {len(valid_split)} images")
    logging.info(f"  Test: {len(test_images)} images")
    
    logging.info("Processing training images...")
    train_copied = 0
    train_failed = 0
    for idx, img_path in enumerate(train_split):
        filename = extract_filename(img_path)
        img_src = os.path.join(source_images_dir, filename)
        label_src = os.path.join(source_images_dir, filename.replace('.jpg', '.txt'))
        
        img_dst = os.path.join(output_dir, 'train', 'images', filename)
        label_dst = os.path.join(output_dir, 'train', 'labels', filename.replace('.jpg', '.txt'))
        
        if copy_file(img_src, img_dst) and copy_file(label_src, label_dst):
            train_copied += 1
        else:
            train_failed += 1
        
        if (idx + 1) % 50 == 0:
            logging.info(f"  Processed {idx + 1}/{len(train_split)} training images...")
    
    logging.info(f"Training: {train_copied} copied, {train_failed} failed")
    
    logging.info("Processing validation images...")
    valid_copied = 0
    valid_failed = 0
    for img_path in valid_split:
        filename = extract_filename(img_path)
        img_src = os.path.join(source_images_dir, filename)
        label_src = os.path.join(source_images_dir, filename.replace('.jpg', '.txt'))
        
        img_dst = os.path.join(output_dir, 'valid', 'images', filename)
        label_dst = os.path.join(output_dir, 'valid', 'labels', filename.replace('.jpg', '.txt'))
        
        if copy_file(img_src, img_dst) and copy_file(label_src, label_dst):
            valid_copied += 1
        else:
            valid_failed += 1
    
    logging.info(f"Validation: {valid_copied} copied, {valid_failed} failed")
    
    logging.info("Processing test images...")
    test_copied = 0
    test_failed = 0
    for img_path in test_images:
        filename = extract_filename(img_path)
        img_src = os.path.join(source_images_dir, filename)
        label_src = os.path.join(source_images_dir, filename.replace('.jpg', '.txt'))
        
        img_dst = os.path.join(output_dir, 'test', 'images', filename)
        label_dst = os.path.join(output_dir, 'test', 'labels', filename.replace('.jpg', '.txt'))
        
        if copy_file(img_src, img_dst) and copy_file(label_src, label_dst):
            test_copied += 1
        else:
            test_failed += 1
    
    logging.info(f"Test: {test_copied} copied, {test_failed} failed")
    
    print("\n" + "="*50)
    print("Restructuration complete!")
    print("="*50)
    print(f"Total images processed:")
    print(f"  Train: {train_copied}/{len(train_split)}")
    print(f"  Validation: {valid_copied}/{len(valid_split)}")
    print(f"  Test: {test_copied}/{len(test_images)}")
    print(f"\nOutput directory: {output_dir}")
    
    if train_failed > 0 or valid_failed > 0 or test_failed > 0:
        logging.warning(f"{train_failed + valid_failed + test_failed} files failed to copy.")
        logging.warning("Please check if all image and label files exist in the source directory.")
    
    return {
        'output_dir': output_dir,
        'train_count': train_copied,
        'valid_count': valid_copied,
        'test_count': test_copied,
        'train_failed': train_failed,
        'valid_failed': valid_failed,
        'test_failed': test_failed
    }

def main():
    """Main function for standalone execution"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    result = reorganize_dataset(
        dataset_path=current_dir,
        validation_split=0.2,
        output_dir_name='traffic_signs'
    )
    
    print("\n" + "="*50)
    print("Reorganization complete!")
    print("="*50)
    print(f"Total images processed:")
    print(f"  Train: {result['train_count']}")
    print(f"  Validation: {result['valid_count']}")
    print(f"  Test: {result['test_count']}")
    print(f"\nOutput directory: {result['output_dir']}")
    
    if result['train_failed'] > 0 or result['valid_failed'] > 0 or result['test_failed'] > 0:
        print(f"\nWarning: {result['train_failed'] + result['valid_failed'] + result['test_failed']} files failed to copy.")

if __name__ == "__main__":
    main()

