from utils.video_utils import load_video, read_frame, release_video
from tracking.player_tracker import PlayerTracker
from detection.yolo_detector import YOLODetector
from tracking.shuttle_tracker import ShuttleTracker
from utils.drawing import draw_tracks, draw_shuttle
from utils.roi_filter import filter_tracks_by_roi, draw_court_roi

import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw_videos", "sample.mp4")

COURT_ROI = [(180, 141), (467, 139), (595, 358), (69, 358)]

def main():
    cap = load_video(VIDEO_PATH)
    player_tracker = PlayerTracker("yolov8n.pt")
    # detector = YOLODetector("yolov8n.pt")       # replace with custom model later
    SHUTTLE_MODEL = os.path.join(BASE_DIR, "models", "weights", "shuttle_best.pt")
    detector = YOLODetector(SHUTTLE_MODEL)
    shuttle_tracker = ShuttleTracker(trail_length=30)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    print(f"FPS: {fps}")

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        # --- Player tracking ---
        tracks = player_tracker.track(frame)
        tracks = filter_tracks_by_roi(tracks, COURT_ROI)

        # --- Shuttle detection + Kalman tracking ---
        _, shuttles = detector.detect(frame)
        shuttle_center = detector.get_best_shuttle(shuttles)
        shuttle_pos = shuttle_tracker.update(shuttle_center)
        trail = shuttle_tracker.get_trail()

        # --- Draw everything ---
        frame = draw_court_roi(frame, COURT_ROI)
        frame = draw_tracks(frame, tracks)
        frame = draw_shuttle(frame, shuttle_pos, trail)

        cv2.imshow("Badminton Analysis", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:
            break
        if cv2.getWindowProperty("Badminton Analysis", cv2.WND_PROP_VISIBLE) < 1:
            break

    release_video(cap)

if __name__ == "__main__":
    main()