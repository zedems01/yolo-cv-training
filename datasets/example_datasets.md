# Dataset

## Using a Kaggle Dataset

To train with a Kaggle dataset, use the `--dataset` argument:
```bash
python scripts/main.py --dataset "jocelyndumlao/multi-weather-pothole-detection-mwpd" --nc 1 --names "Potholes"
```

## Using a Local Dataset

### Option 1: Standard Format

Organize your dataset following a commom YOLO structure, like:

```
my_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/  (optional)
    ├── images/
    └── labels/
```
Then, use the `--local-dataset` argument:
```bash
python scripts/main.py --local-dataset "C:\path\to\my_dataset" --nc 2 --names "Class1,Class2"
```
The script will automatically detect the structure and generate the required `data.yaml` file.

### Option 2: Custom Format with data.yaml

If your dataset has a valid but custom structure, you can manually create a `data.yaml` file:

```yaml
path: C:\path\to\my_dataset  # Absolute or relative path to the root folder
train: train/images  # Relative path from 'path' to training images
val: valid/images    # Relative path from 'path' to validation images
test: test/images    # Relative path from 'path' to test images (optional)

nc: 2  # Number of classes
names: ['Class1', 'Class2']  # List of class names
```

### Option 3: Reorganizing with a Script

If your dataset is not in the standard format, you can create a reorganization script to match a common format.

*Example with the `valentynsichkar/traffic-signs-dataset-in-yolo-format` dataset :*

```bash
# 1. Download or place your dataset in a folder
# 2. Run the reorganization script from the root folder
python reorganize_traffic_signs_dataset.py

# 3. Use the reorganized folder with --local-dataset
python scripts/main.py --local-dataset "C:\path\to\datasets\traffic_signs" --nc 4 --names "prohibitory,danger,mandatory,other"
```

# Example Kaggle Datasets for YOLO Training

This file contains example Kaggle datasets suitable for training YOLO object detection models. Each dataset includes the handle, number of classes, and class names.
The datasets structure may vary and require adjustments (directly modifying the structure or the `data.yaml` file).

## 1. Pothole Detection
- **Handle**: `jocelyndumlao/multi-weather-pothole-detection-mwpd`
- **Classes**: 1
- **Names**: "Potholes"
- **Description**: Images of potholes and cracks in roads under various weather conditions

## 2. Traffic Signs
- **Handle**: `valentynsichkar/traffic-signs-dataset-in-yolo-format`
- **Classes**: 4
- **Names**: "prohibitory,danger,mandatory,other"
- **Description**: German traffic sign detection dataset with YOLO annotations

## 3. Face Detection
- **Handle**: `jessicali9530/lfw-dataset`
- **Classes**: 1
- **Names**: "face"
- **Description**: Labeled Faces in the Wild dataset for face detection

## 4. Animal Detection
- **Handle**: `antoreepjana/animals-detection-images-dataset`
- **Classes**: 80
- **Names**: "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"
- **Description**: COCO-style animal detection dataset with various species

## 5. Medical X-Ray
- **Handle**: `paultimothymooney/chest-xray-pneumonia`
- **Classes**: 2
- **Names**: "normal,pneumonia"
- **Description**: Chest X-ray images for pneumonia detection

## 6. License Plate Detection
- **Handle**: `andrewmvd/car-plate-detection`
- **Classes**: 1
- **Names**: "license_plate"
- **Description**: Images of cars with license plate annotations for detection

