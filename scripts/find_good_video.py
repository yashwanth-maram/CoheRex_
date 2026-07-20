import os
import cv2
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG
from coherex.detection.yolo_detector import YOLODetector

def main():
    directory = os.path.join(ROOT, "data", "evaluation", "tampered")
    videos = [v for v in os.listdir(directory) if v.endswith(".mp4")][:20]
    
    detector = YOLODetector(config=CONFIG)
    
    for v in videos:
        path = os.path.join(directory, v)
        cap = cv2.VideoCapture(path)
        tracked_frames = 0
        total_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            total_frames += 1
            if total_frames % 5 == 0: # sample every 5th frame for speed
                dets = detector.detect(frame)
                has_person = any(d["class_id"] == CONFIG.detection.person_class_id for d in dets)
                if has_person:
                    tracked_frames += 5
        cap.release()
        
        ratio = tracked_frames / (total_frames + 1e-5)
        print(f"{v}: {tracked_frames}/{total_frames} frames ({ratio*100:.1f}%) have people.")
        
        if ratio > 0.8:
            print(f"--> FOUND GOOD VIDEO: {v}")
            break

if __name__ == "__main__":
    main()
