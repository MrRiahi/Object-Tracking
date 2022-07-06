import cv2


class KCFTracker:

    def __init__(self):
        self.tracker = None

    def kcf_tracker_init(self, frame, bbox):
        """
        Initialize the KCF tracker.
        :param frame: first frame for initialization.
        :param bbox: the first bounding box of the object.
        :return:
        """

        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)

    def kcf_tracker_update(self, frame):
        """
        Track the object in the frame and return its bounding box.
        :param frame: first frame for initialization.
        :return:
        """

        ret, bbox = self.tracker.update(frame)

        return ret, bbox
