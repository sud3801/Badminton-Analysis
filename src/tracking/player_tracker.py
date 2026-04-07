from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def track(self, frame):
        """
        Returns list of tracked players with persistent IDs.
        Each entry: {track_id, bbox, confidence}
        """
        results = self.model.track(
            frame,
            persist=True,       # maintains ID across frames
            classes=[0],        # class 0 = person
            conf=0.4,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        tracks = []
        if results.boxes.id is None:
            return tracks  # no detections this frame

        for box, track_id in zip(results.boxes, results.boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            tracks.append({
                "track_id": int(track_id),
                "bbox": (x1, y1, x2, y2),
                "confidence": conf
            })

        return tracks