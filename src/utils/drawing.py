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

def draw_rally_status(frame, rally_status):
    """
    Draws rally state as an overlay on the top-left of the frame.
    """
    import cv2

    in_rally = rally_status.get("in_rally", False)
    rally_id = rally_status.get("rally_id")
    event    = rally_status.get("event")

    # Flash green on START, red on END
    if event == "START":
        overlay_color = (0, 255, 0)
        label = f"RALLY {rally_id} STARTED"
    elif event == "END":
        overlay_color = (0, 0, 255)
        label = f"RALLY ENDED"
    elif in_rally:
        overlay_color = (0, 200, 0)
        label = f"RALLY {rally_id} IN PROGRESS"
    else:
        overlay_color = (60, 60, 60)
        label = "NO RALLY"

    # Semi-transparent background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (380, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Status text
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, overlay_color, 2)

    # Frame counter (small, bottom right)
    h, w = frame.shape[:2]
    frame_text = f"Frame: {rally_status.get('frame', 0)}"
    cv2.putText(frame, frame_text, (w - 160, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame

def draw_stable_players(frame, players):
    """Draw stable Player A / Player B with consistent colors."""
    import cv2

    for p in players:
        if "bbox" not in p:
            continue

        x1, y1, x2, y2 = p["bbox"]
        color  = p["color"]
        label  = f"Player {p['label']}  ({p['confidence']:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Foot point dot
        fx, fy = p["foot"]
        cv2.circle(frame, (fx, fy), 5, color, -1)

        # Show if player is lost (predicting)
        if p.get("lost_frames", 0) > 0:
            cv2.putText(frame, "lost", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    return frame