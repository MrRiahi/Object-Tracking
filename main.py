import cv2
import time
import numpy as np

from tracker import MILTracker, KCFTracker, CSRTTracker, MOSSETacker, TLDTracker, \
                    BoostingTracker, DaSiamRPNTacker, GOTURNTacker
from utils import save_video

# Initialize video
video = cv2.VideoCapture('input_data/car-overhead-1_cut.avi')
bbox = (195, 121, 26, 21)

if not video.isOpened():
    raise Exception('Invalid video')

# Read first frame
ret, frame = video.read()

# Initialize tracker
tracker = DaSiamRPNTacker()
tracker.dasiamrpn_tracker_init(frame=frame, bbox=bbox)

frames = []
times = []
while True:
    ret, frame = video.read()

    if not ret:
        break

    tic = time.time()
    ret, bbox = tracker.dasiamrpn_tracker_update(frame=frame)
    toc = time.time()
    times.append(toc - tic)

    cv2.putText(frame, str(round(1000 * (toc - tic), 2)) + ' ms', (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Draw bounding box
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    frames.append(frame)


print('------------number of frames:', len(times))
print('----------times:', times)
print('------------average_time:', np.mean(times))
save_video(frames=frames, video_name='car-overhead-1_cut_DaSiamRPNTacker.avi')

