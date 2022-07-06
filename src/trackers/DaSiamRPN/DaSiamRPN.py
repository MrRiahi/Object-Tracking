import cv2


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
