import cv2

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
    return cap

def read_frame(cap):
    ret, frame = cap.read()
    return ret, frame

def release_video(cap):
    cap.release()
    cv2.destroyAllWindows()