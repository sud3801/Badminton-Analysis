# src/detection/yolo_detector.py
from ultralytics import YOLO

# Update these class IDs based on your trained model
# For now using COCO defaults (person=0); shuttle needs custom model
PERSON_CLASS_ID = 0
SHUTTLE_CLASS_ID = 0  # update after training custom model

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        players = []
        shuttles = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            det = {
                "class_id": cls_id,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2)
            }

            if cls_id == PERSON_CLASS_ID and conf > 0.5:
                players.append(det)
            elif cls_id == SHUTTLE_CLASS_ID and conf > 0.3:
                shuttles.append(det)

        return players, shuttles

    def get_best_shuttle(self, shuttles):
        """Return the highest-confidence shuttle detection center."""
        if not shuttles:
            return None
        best = max(shuttles, key=lambda d: d["confidence"])
        return best["center"]