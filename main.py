import cv2
import time
import numpy as np

from src.config import Config as Cfg
from src.utils import save_video
from src.tracker import get_tracker

# Initializations
video_name = 'sample_1.avi'
video_path = f'input_data/{video_name}'
output_path = f'output_data/{video_path.split(".")[0]}_{Cfg.TRACKER_TYPE}.avi'

# Initialize video
video = cv2.VideoCapture(video_name)
bbox = (195, 121, 26, 21)

if not video.isOpened():
    raise Exception('Invalid video')

# Read first frame
ret, frame = video.read()

# Get tracker
tracker = get_tracker(frame=frame, bbox=bbox)

frames = []
while True:
    ret, frame = video.read()

    if not ret:
        break

    tic = time.time()
    ret, bbox = tracker.dasiamrpn_tracker_update(frame=frame)
    toc = time.time()

    cv2.putText(frame, str(round(1000 * (toc - tic), 2)) + ' ms', (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Draw bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    frames.append(frame)

save_video(frames=frames, video_name=output_path)

