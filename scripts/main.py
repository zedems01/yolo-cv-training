import argparse
import logging
import os
import yaml
import kagglehub
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_dataset(dataset_handle):
    """Download the Kaggle dataset."""
    try:
        path = kagglehub.dataset_download(dataset_handle)
        logging.info(f"Downloaded dataset to: {path}")
        return path
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise

def detect_dataset_structure(dataset_path):
    """Detect train/val/test images and labels paths in the dataset folder."""
    paths = {}
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    logging.info(f"Detected subdirs in dataset: {subdirs}")
    splits = ['train', 'valid', 'test']
    for split in splits:
        key_split = 'val' if split == 'valid' else split
        if split in subdirs:
            split_dir = os.path.join(dataset_path, split)
            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'labels')
            if os.path.exists(images_dir):
                paths[f'{key_split}_images'] = images_dir
            if os.path.exists(labels_dir):
                paths[f'{key_split}_labels'] = labels_dir
    if not paths:
        # Check one level deeper
        for subdir in subdirs:
            sub_path = os.path.join(dataset_path, subdir)
            if os.path.isdir(sub_path):
                inner_subdirs = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
                logging.info(f"Checking inner subdirs in {subdir}: {inner_subdirs}")
                for split in splits:
                    key_split = 'val' if split == 'valid' else split
                    if split in inner_subdirs:
                        split_dir = os.path.join(sub_path, split)
                        images_dir = os.path.join(split_dir, 'images')
                        labels_dir = os.path.join(split_dir, 'labels')
                        if os.path.exists(images_dir):
                            paths[f'{key_split}_images'] = images_dir
                        if os.path.exists(labels_dir):
                            paths[f'{key_split}_labels'] = labels_dir
                if paths:
                    dataset_path = sub_path
                    break
        if not paths:
            # Check two levels deeper
            for subdir in subdirs:
                sub_path = os.path.join(dataset_path, subdir)
                if os.path.isdir(sub_path):
                    inner_subdirs = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
                    for inner_subdir in inner_subdirs:
                        inner_path = os.path.join(sub_path, inner_subdir)
                        if os.path.isdir(inner_path):
                            deepest_subdirs = [d for d in os.listdir(inner_path) if os.path.isdir(os.path.join(inner_path, d))]
                            logging.info(f"Checking deepest subdirs in {inner_subdir}: {deepest_subdirs}")
                            for split in splits:
                                key_split = 'val' if split == 'valid' else split
                                if split in deepest_subdirs:
                                    split_dir = os.path.join(inner_path, split)
                                    images_dir = os.path.join(split_dir, 'images')
                                    labels_dir = os.path.join(split_dir, 'labels')
                                    if os.path.exists(images_dir):
                                        paths[f'{key_split}_images'] = images_dir
                                    if os.path.exists(labels_dir):
                                        paths[f'{key_split}_labels'] = labels_dir
                            if paths:
                                dataset_path = inner_path
                                break
                    if paths:
                        break
    logging.info(f"Detected paths: {paths}")
    return paths, dataset_path

def create_yaml(dataset_path, paths, nc, names):
    """Create the data.yaml file for YOLO training."""
    data_yaml = {
        "path": dataset_path,
        "train": os.path.relpath(paths.get('train_images', ''), dataset_path) if 'train_images' in paths else '',
        "val": os.path.relpath(paths.get('val_images', ''), dataset_path) if 'val_images' in paths else '',
        "test": os.path.relpath(paths.get('test_images', ''), dataset_path) if 'test_images' in paths else '',
        "nc": nc,
        "names": names,
    }
    yaml_path = f"{os.path.basename(dataset_path)}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    logging.info(f"Created YAML config at: {yaml_path}")
    return yaml_path
