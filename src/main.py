from utils.video_utils import load_video, read_frame, release_video
from tracking.player_tracker import PlayerTracker
from tracking.player_manager import PlayerManager
from detection.yolo_detector import YOLODetector
from tracking.shuttle_tracker import ShuttleTracker
from utils.drawing import draw_stable_players, draw_shuttle
from utils.roi_filter import filter_tracks_by_roi, draw_court_roi

import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw_videos", "INDvsDEN.mp4")

COURT_ROI = [(429, 234), (926, 234), (1172, 713), (201, 712)]

xs = [p[0] for p in COURT_ROI]
ys = [p[1] for p in COURT_ROI]
COURT_BOUNDS = (min(xs), min(ys), max(xs), max(ys))


def main():
    cap = load_video(VIDEO_PATH)
    player_tracker = PlayerTracker("yolov8s.pt")
    # detector = YOLODetector("yolov8n.pt")       # replace with custom model later
    SHUTTLE_MODEL = os.path.join(BASE_DIR, "models", "weights", "shuttle_best.pt")
    detector = YOLODetector(SHUTTLE_MODEL)
    shuttle_tracker = ShuttleTracker(trail_length=30)
    player_manager = PlayerManager(court_roi=COURT_ROI, reidentify_dist=180) # adjust reidentify_dist as needed

    DESIRED_FPS = 25  # ← Change this value to your desired FPS
    fps = DESIRED_FPS
    delay = int(1000 / fps)
    print(f"Display FPS: {fps} (delay: {delay}ms)")

    PROCESS_WIDTH = 1280

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        scale = PROCESS_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (PROCESS_WIDTH, int(frame.shape[0] * scale)))

        # --- Player tracking ---
        raw_tracks = player_tracker.track(frame)
        raw_tracks = filter_tracks_by_roi(raw_tracks, COURT_ROI)
        players = player_manager.update(raw_tracks)

        # --- Shuttle detection + Kalman tracking ---
        _, shuttles = detector.detect(frame)
        shuttle_center = detector.get_best_shuttle(shuttles)
        shuttle_pos = shuttle_tracker.update(shuttle_center)
        trail = shuttle_tracker.get_trail()

        # --- Draw everything ---
        frame = draw_court_roi(frame, COURT_ROI)
        #frame = draw_tracks(frame, tracks)
        frame = draw_stable_players(frame, players) 
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