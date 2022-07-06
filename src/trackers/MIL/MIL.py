import cv2


class MILTracker:

    def __init__(self):
        self.tracker = None

    def mil_tracker_init(self, frame, bbox):
        """
        Initialize the MIL tracker.
        :param frame: first frame for initialization.
        :param bbox: the first bounding box of the object.
        :return:
        """

        self.tracker = cv2.TrackerMIL_create()
        self.tracker.init(frame, bbox)

    def mil_tracker_update(self, frame):
        """
        Track the object in the frame and return its bounding box.
        :param frame: first frame for initialization.
        :return:
        """

        ret, bbox = self.tracker.update(frame)

        return ret, bbox
