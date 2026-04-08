import cv2

# Color palette for different track IDs
COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 100, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0)
]

def draw_detections(frame, detections):
    """Draw raw detections (no ID)."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        label = f"cls:{det['class_id']} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def draw_tracks(frame, tracks):

    """Draw tracked players with persistent colored IDs."""
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid = t["track_id"]
        color = COLORS[tid % len(COLORS)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"P{tid} ({t['confidence']:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw center dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame
def draw_shuttle(frame, position, trail, radius=6):
    """
    Draw shuttle position and fading trajectory trail.
    - Bright circle at current position
    - Fading colored trail showing recent path
    """
    import cv2
    import numpy as np

    n = len(trail)
    for i in range(1, n):
        if trail[i - 1] is None or trail[i] is None:
            continue

        # Fade from dim to bright yellow as trail gets more recent
        alpha = i / n
        color = (0, int(255 * alpha), int(255 * alpha))
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, trail[i - 1], trail[i], color, thickness)

    # Draw current position
    if position:
        cv2.circle(frame, position, radius, (0, 255, 255), -1)       # filled circle
        cv2.circle(frame, position, radius + 2, (0, 180, 255), 2)    # outer ring

    return frame