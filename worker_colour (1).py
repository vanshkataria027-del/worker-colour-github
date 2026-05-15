import cv2
import numpy as np
import argparse
from ultralytics import YOLO


def detect_uniform(video_source, model_path="yolov8n.pt"):
    """
    Detect workers in a video and check if they are wearing blue uniforms.

    Args:
        video_source: Path to video file or camera index (0 for webcam)
        model_path:   Path to YOLO model weights file
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_source}")
        return

    print("[INFO] Detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect only persons (class 0), suppress verbose logs
        results = model(frame, classes=[0], verbose=False)[0]

        blue_count = 0
        no_blue_count = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            if confidence < 0.4:
                continue

            person_crop = hsv[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Blue HSV range
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(person_crop, lower_blue, upper_blue)

            if mask.sum() > 5000:
                blue_count += 1
                color = (255, 0, 0)
                label = f"Blue Uniform {confidence:.2f}"
            else:
                no_blue_count += 1
                color = (0, 0, 255)
                label = f"No Uniform {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.putText(frame, f"Blue: {blue_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"No Blue: {no_blue_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Worker Uniform Colour Detector")
    p.add_argument("--video", required=True,
                   help="Path to video file, or 0 for webcam")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO model path (default: yolov8n.pt)")
    args = p.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    detect_uniform(source, args.model)
