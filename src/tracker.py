import cv2


# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
class CSRTTracker:

    def __init__(self):
        self.tracker = None

    def csrt_tracker_init(self, frame, bbox):
        """
        Initialize the CSRT tracker
        :param frame: first frame for initialization.
        :param bbox: the first bounding box of the object.
        :return:
        """

        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)

    def csrt_tracker_update(self, frame):
        """
        Track the object in the frame and return its bounding box.
        :param frame: first frame for initialization.
        :return:
        """

        ret, bbox = self.tracker.update(frame)

        return ret, bbox


# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
class TLDTracker:

    def __init__(self):
        self.tracker = None

    def tld_tracker_init(self, frame, bbox):
        """
        Initialize the TLD tracker
        :param frame: first frame for initialization.
        :param bbox: the first bounding box of the object.
        :return:
        """

        self.tracker = cv2.legacy.TrackerTLD_create()
        self.tracker.init(frame, bbox)

    def tld_tracker_update(self, frame):
        """
        Track the object in the frame and return its bounding box.
        :param frame: first frame for initialization.
        :return:
        """

        ret, bbox = self.tracker.update(frame)

        return ret, bbox


# -------------------------------------------------------------------------
class BoostingTracker:

    def __init__(self):
        self.tracker = None

    def boosting_tracker_init(self, frame, bbox):
        """
        Initialize the Boosting tracker
        :param frame: first frame for initialization.
        :param bbox: the first bounding box of the object.
        :return:
        """

        self.tracker = cv2.legacy.TrackerBoosting_create()
        self.tracker.init(frame, bbox)

    def boosting_tracker_update(self, frame):
        """
        Track the object in the frame and return its bounding box.
        :param frame: first frame for initialization.
        :return:
        """

        ret, bbox = self.tracker.update(frame)

        return ret, bbox


# -------------------------------------------------------------------------
class DaSiamRPNTacker:

    def __init__(self):
        self.tracker = None
        self.params = cv2.TrackerDaSiamRPN_Params()
        self.params.model = './models/dasiamrpn/dasiamrpn_model.onnx'
        self.params.kernel_cls1 = './models/dasiamrpn/dasiamrpn_kernel_cls1.onnx'
        self.params.kernel_r1 = './models/dasiamrpn/dasiamrpn_kernel_r1.onnx'

    def dasiamrpn_tracker_init(self, frame, bbox):
        """
        Initialize the DaSiamRPN tracker
        :param frame: first frame for initialization.
        :param bbox: the first bounding box of the object.
        :return:
        """

        self.tracker = cv2.TrackerDaSiamRPN_create(self.params)
        self.tracker.init(frame, bbox)

    def dasiamrpn_tracker_update(self, frame):
        """
        Track the object in the frame and return its bounding box.
        :param frame: first frame for initialization.
        :return:
        """

        ret, bbox = self.tracker.update(frame)

        return ret, bbox


# -------------------------------------------------------------------------
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
