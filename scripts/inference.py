import argparse
import logging
import cv2
import os
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model(model_path):
    """Load the YOLO model from the specified path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)
    logging.info(f"Model loaded from {model_path}")
    return model


def infer_image(model, image_path, conf_thresh=0.5, save_path=None):
    """Perform inference on a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    results = model(img, conf=conf_thresh)
    annotated = results[0].plot()
    if save_path:
        cv2.imwrite(save_path, annotated)
        logging.info(f"Annotated image saved to {save_path}")
    else:
        cv2.imshow("Inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def infer_video(model, video_path, conf_thresh=0.5, save_path=None):
    """Perform inference on a video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_thresh)
        annotated = results[0].plot()
        if save_path:
            out.write(annotated)
        else:
            cv2.imshow("Inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_count += 1
        if frame_count % 100 == 0:
            logging.info(f"Processed {frame_count} frames")
    cap.release()
    if save_path:
        out.release()
        logging.info(f"Annotated video saved to {save_path}")
    cv2.destroyAllWindows()


def infer_webcam(model, conf_thresh=0.5):
    """Perform real-time inference on webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot access webcam")
    logging.info("Starting webcam inference. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_thresh)
        annotated = results[0].plot()
        cv2.imshow("Webcam Inference", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def infer_folder(model, input_folder, output_folder, conf_thresh=0.5):
    """Perform inference on all images in a folder."""
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input path is not a directory: {input_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        raise ValueError(f"No image files found in {input_folder}")
    logging.info(f"Found {len(image_files)} images in {input_folder}")
    
    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        try:
            img = cv2.imread(input_path)
            if img is None:
                logging.warning(f"Could not read image: {input_path}")
                continue
            results = model(img, conf=conf_thresh)
            annotated = results[0].plot()
            cv2.imwrite(output_path, annotated)
            if idx % 10 == 0 or idx == len(image_files):
                logging.info(f"Processed {idx}/{len(image_files)} images")
        except Exception as e:
            logging.error(f"Error processing {input_path}: {e}")
            continue
    
    logging.info(f"All images processed. Results saved to {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model weights, e.g., runs/train/weights/best.pt",
    )
    parser.add_argument(
        "--input", help='Input: image/video path or "webcam"'
    )
    parser.add_argument(
        "--input-folder", help="Input folder containing test images"
    )
    parser.add_argument(
        "--output-folder", help="Output folder for saving annotated images (required when using --input-folder)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument("--output", help="Output path for saving results (optional)")
    args = parser.parse_args()

    try:
        model = load_model(args.model)
        
        # Handle folder inference
        if args.input_folder:
            if not args.output_folder:
                raise ValueError("--output-folder is required when using --input-folder")
            infer_folder(model, args.input_folder, args.output_folder, args.conf)
        # Handle single file or webcam inference
        elif args.input:
            input_lower = args.input.lower()
            if input_lower == "webcam":
                infer_webcam(model, args.conf)
            elif input_lower.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
                infer_image(model, args.input, args.conf, args.output)
            elif input_lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".flv")):
                infer_video(model, args.input, args.conf, args.output)
            else:
                raise ValueError("Unsupported input type. Use image/video path or 'webcam'")
        else:
            raise ValueError("Either --input or --input-folder must be provided")
        
        logging.info("Inference completed successfully")
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()

