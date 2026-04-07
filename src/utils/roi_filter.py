# src/utils/roi_filter.py
import cv2
import numpy as np

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon."""
    polygon_np = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon_np, point, False)
    return result >= 0  # >= 0 means inside or on edge

def filter_tracks_by_roi(tracks, court_roi):
    """
    Keep only tracks whose bounding box CENTER is inside the court ROI.
    
    Args:
        tracks: list of track dicts with 'bbox'
        court_roi: list of (x, y) points defining court polygon
    Returns:
        filtered list of tracks
    """
    if not court_roi:
        return tracks  # no ROI defined, return all

    filtered = []
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        # Use BOTTOM-CENTER of bbox (feet position — more accurate for players)
        foot_x = (x1 + x2) // 2
        foot_y = y2  # bottom of bounding box

        if point_in_polygon((float(foot_x), float(foot_y)), court_roi):
            filtered.append(t)

    return filtered

def draw_court_roi(frame, court_roi, color=(0, 255, 255), thickness=2):
    """Visualize the court boundary on frame."""
    if not court_roi:
        return frame
    pts = np.array(court_roi, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    return frame