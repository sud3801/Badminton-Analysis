# src/utils/roi_selector.py
import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(param, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Select Court ROI", param)

def select_roi(video_path):
    """
    Click the 4 corners of the court (in order: top-left, top-right,
    bottom-right, bottom-left). Press 'q' when done.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Could not read video")

    clone = frame.copy()
    cv2.imshow("Select Court ROI", clone)
    cv2.setMouseCallback("Select Court ROI", click_event, clone)

    print("Click the 4 corners of the court. Press 'q' when done.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Court ROI points:", points)
    return points

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw_videos", "sample.mp4")
    pts = select_roi(VIDEO_PATH)
    print("\nCopy this into your config:\nCOURT_ROI =", pts)