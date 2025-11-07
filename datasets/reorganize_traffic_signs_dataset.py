# Reorganize 'valentynsichkar/traffic-signs-dataset-in-yolo-format' dataset with train/valid/test folders
# 

import os
import shutil
import random
from pathlib import Path

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

def main():
    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_images_dir = os.path.join(current_dir, 'ts', 'ts')
    train_txt = os.path.join(current_dir, 'train.txt')
    test_txt = os.path.join(current_dir, 'test.txt')
    output_dir = os.path.join(current_dir, 'traffic_signs')
    validation_split = 0.2
    random.seed(42)
    
    print("Starting dataset restructuration...")
    print(f"Source directory: {source_images_dir}")
    print(f"Output directory: {output_dir}")
    
    print("\nCreating directory structure...")
    create_directory_structure(output_dir)
    
    # Read image lists
    print("\nReading image lists...")
    train_images = read_image_list(train_txt)
    test_images = read_image_list(test_txt)
    
    print(f"Training images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    
    # Split training data into train and validation
    random.shuffle(train_images)
    split_idx = int(len(train_images) * (1 - validation_split))
    train_split = train_images[:split_idx]
    valid_split = train_images[split_idx:]
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_split)} images")
    print(f"  Validation: {len(valid_split)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Process training images
    print("\nProcessing training images...")
    train_copied = 0
    train_failed = 0
    for img_path in train_split:
        filename = extract_filename(img_path)
        img_src = os.path.join(source_images_dir, filename)
        label_src = os.path.join(source_images_dir, filename.replace('.jpg', '.txt'))
        
        img_dst = os.path.join(output_dir, 'train', 'images', filename)
        label_dst = os.path.join(output_dir, 'train', 'labels', filename.replace('.jpg', '.txt'))
        
        if copy_file(img_src, img_dst) and copy_file(label_src, label_dst):
            train_copied += 1
        else:
            train_failed += 1
            if train_copied % 50 == 0:
                print(f"  Processed {train_copied} images...")
    
    print(f"  Training: {train_copied} copied, {train_failed} failed")
    
    # Process validation images
    print("\nProcessing validation images...")
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
    
    print(f"  Validation: {valid_copied} copied, {valid_failed} failed")
    
    # Process test images
    print("\nProcessing test images...")
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
    
    print(f"  Test: {test_copied} copied, {test_failed} failed")
    
    # Summary
    print("\n" + "="*50)
    print("Restructuration complete!")
    print("="*50)
    print(f"Total images processed:")
    print(f"  Train: {train_copied}/{len(train_split)}")
    print(f"  Validation: {valid_copied}/{len(valid_split)}")
    print(f"  Test: {test_copied}/{len(test_images)}")
    print(f"\nOutput directory: {output_dir}")
    
    if train_failed > 0 or valid_failed > 0 or test_failed > 0:
        print(f"\nWarning: {train_failed + valid_failed + test_failed} files failed to copy.")
        print("Please check if all image and label files exist in the source directory.")

if __name__ == "__main__":
    main()

