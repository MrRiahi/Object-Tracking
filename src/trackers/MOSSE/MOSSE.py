import cv2


class MOSSETacker:

    def __init__(self):
        self.tracker = None

    def mosse_tracker_init(self, frame, bbox):
        """
        Initialize the MOSSE tracker
        :param frame: first frame for initialization.
        :param bbox: the first bounding box of the object.
        :return:
        """

        self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.tracker.init(frame, bbox)

    def mosse_tracker_update(self, frame):
        """
        Track the object in the frame and return its bounding box.
        :param frame: first frame for initialization.
        :return:
        """

        ret, bbox = self.tracker.update(frame)

        return ret, bbox
