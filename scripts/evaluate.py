import argparse
import json
import logging
import os
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_model(model_path, data_yaml, split='test', imgsz=640, conf=0.25):
    """Evaluate a trained YOLO model on test dataset."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
    
    model_dir = os.path.dirname(model_path)
    train_dir = os.path.dirname(model_dir)
    project_dir = os.path.dirname(train_dir)
    name = os.path.basename(train_dir)
    output_dir = f"{name}/{split}"

    logging.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    logging.info(f"Evaluating on {split} split...")
    logging.info(f"Data YAML: {data_yaml}")
    logging.info(f"Results will be saved to: {output_dir}")
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        plots=True,
        project=project_dir,
        name=f"{output_dir}",
    )

    metrics = {
        "split": split,
        "mAP50": round(results.box.map50, 3),
        "mAP50-95": round(results.box.map, 3),
        "precision": round(results.box.mp, 3),
        "recall": round(results.box.mr, 3),
        "conf_threshold": conf,
        "imgsz": imgsz,
    }
    res_dir = os.path.join(train_dir, split, "results.json")
    with open(res_dir, "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info("="*70)
    logging.info("Evaluation Results:")
    logging.info("="*70)
    logging.info(f"mAP50: {metrics['mAP50']:.3f}")
    logging.info(f"mAP50-95: {metrics['mAP50-95']:.3f}")
    logging.info(f"Precision: {metrics['precision']:.3f}")
    logging.info(f"Recall: {metrics['recall']:.3f}")
    logging.info("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model on test dataset")
    parser.add_argument('--model', required=True, help='Path to trained model weights, e.g., runs/train/yolo_train/weights/best.pt')
    parser.add_argument('--data', required=True, help='Path to data.yaml file (must contain test split)')
    parser.add_argument('--split', default='test', choices=['test', 'val', 'train'], help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for evaluation (default: 640)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for evaluation (default: 0.25)')
    args = parser.parse_args()
    
    try:
        results = evaluate_model(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
        )
        logging.info("Evaluation completed successfully")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

