import cv2


class GOTURNTacker:

    def __init__(self):
        self.tracker = None
        self.params = cv2.TrackerGOTURN_Params()
        self.params.modelTxt = './models/goturn/goturn.prototxt'
        self.params.modelBin = './models/goturn/goturn.caffemodel'

    def goturn_tracker_init(self, frame, bbox):
        """
        Initialize the GOTURN tracker
        :param frame: first frame for initialization.
        :param bbox: the first bounding box of the object.
        :return:
        """

        self.tracker = cv2.TrackerGOTURN_create(self.params)
        self.tracker.init(frame, bbox)

    def goturn_tracker_update(self, frame):
        """
        Track the object in the frame and return its bounding box.
        :param frame: first frame for initialization.
        :return:
        """

        ret, bbox = self.tracker.update(frame)

        return ret, bbox
