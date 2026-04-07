from utils.video_utils import load_video, read_frame, release_video
from detection.yolo_detector import YOLODetector
from utils.drawing import draw_detections

import cv2
import os



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw_videos", "sample.mp4")

def main():
    cap = load_video(VIDEO_PATH)

    detector = YOLODetector("yolov8n.pt")

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    print("FPS:", fps)

    while True:
        ret, frame = read_frame(cap)
        
        if not ret:
            break

        detections = detector.detect(frame)
        detections = [d for d in detections if d["class_id"] == 0 and d["confidence"] > 0.5]

        frame = draw_detections(frame, detections)

        cv2.imshow("Detection", frame)

        key = cv2.waitKey(delay) & 0xFF

        if key == 27:  # ESC key
            break

        if cv2.getWindowProperty("Detection", cv2.WND_PROP_VISIBLE) < 1:
            break


    release_video(cap)

if __name__ == "__main__":
    main()