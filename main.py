import cv2
# import numpy as np
# import matplotlib.pyplot as plt

from tracker import MILTracker, KCFTracker, CSRTTracker, DaSiamRPNTacker
from utils import save_video

# Initialize video
video = cv2.VideoCapture('input_data/car-overhead-1_cut.avi')
bbox = (195, 121, 26, 21)

if not video.isOpened():
    raise Exception('Invalid video')

# Read first frame
ret, frame = video.read()

# Initialize tracker
tracker = CSRTTracker()
ret = tracker.csrt_tracker_init(frame=frame, bbox=bbox)

frames = []
while True:
    ret, frame = video.read()

    if not ret:
        break

    ret, bbox = tracker.csrt_tracker_update(frame=frame)

    # Draw bounding box
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    frames.append(frame)


save_video(frames=frames, video_name='car-overhead-1_cut_dasiamrpn_tracker.avi')

