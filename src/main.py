from utils.video_utils import load_video, read_frame, release_video
from tracking.player_tracker import PlayerTracker
from utils.drawing import draw_tracks
from utils.roi_filter import filter_tracks_by_roi, draw_court_roi

import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw_videos", "sample.mp4")

# ✏️ Paste your court ROI points here after running roi_selector.py
# Example (replace with your actual values):

COURT_ROI = [(182, 147), (468, 148), (588, 357), (68, 354)]

def main():
    cap = load_video(VIDEO_PATH)
    tracker = PlayerTracker("yolov8n.pt")

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    print(f"FPS: {fps}")

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        # Detect + track all persons
        tracks = tracker.track(frame)

        # Filter to court region only
        tracks = filter_tracks_by_roi(tracks, COURT_ROI)

        # Draw court boundary + filtered tracks
        frame = draw_court_roi(frame, COURT_ROI)
        frame = draw_tracks(frame, tracks)

        cv2.imshow("Player Tracking", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:
            break
        if cv2.getWindowProperty("Player Tracking", cv2.WND_PROP_VISIBLE) < 1:
            break

    release_video(cap)

if __name__ == "__main__":
    main()