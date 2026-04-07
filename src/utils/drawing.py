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