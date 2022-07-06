from src.trackers.DaSiamRPN.DaSiamRPN import DaSiamRPNTacker
from src.trackers.Boosting.Boosting import BoostingTracker
from src.trackers.GOTURN.GOTURN import GOTURNTacker
from src.trackers.MOSSE.MOSSE import MOSSETacker
from src.trackers.CSRT.CSRT import CSRTTracker
from src.trackers.MIL.MIL import MILTracker
from src.trackers.KCF.KCF import KCFTracker
from src.trackers.TLD.TLD import TLDTracker

from src.config import Config as Cfg


def get_tracker(frame, bbox):
    """
    Load the tracker and initialize it
    :param frame: the first frame to initialize the bbox on it.
    :param bbox: bbox of the object.
    :return:
    """

    # MILTracker, KCFTracker, CSRTTracker, MOSSETacker, TLDTracker, BoostingTracker,
    # DaSiamRPNTacker, GOTURNTacker
    if Cfg.TRACKER_TYPE == 'MILTracker':
        tracker = MILTracker()
        tracker.mil_tracker_init(frame=frame, bbox=bbox)

    elif Cfg.TRACKER_TYPE == 'KCFTracker':
        tracker = KCFTracker()
        tracker.kcf_tracker_init(frame=frame, bbox=bbox)

    elif Cfg.TRACKER_TYPE == 'CSRTTracker':
        tracker = CSRTTracker()
        tracker.csrt_tracker_init(frame=frame, bbox=bbox)

    elif Cfg.TRACKER_TYPE == 'MOSSETacker':
        tracker = MOSSETacker()
        tracker.mosse_tracker_init(frame=frame, bbox=bbox)

    elif Cfg.TRACKER_TYPE == 'TLDTracker':
        tracker = TLDTracker()
        tracker.tld_tracker_init(frame=frame, bbox=bbox)

    elif Cfg.TRACKER_TYPE == 'BoostingTracker':
        tracker = BoostingTracker()
        tracker.boosting_tracker_init(frame=frame, bbox=bbox)

    elif Cfg.TRACKER_TYPE == 'DaSiamRPNTacker':
        tracker = DaSiamRPNTacker()
        tracker.dasiamrpn_tracker_init(frame=frame, bbox=bbox)

    elif Cfg.TRACKER_TYPE == 'GOTURNTacker':
        tracker = GOTURNTacker()
        tracker.goturn_tracker_init(frame=frame, bbox=bbox)

    else:
        raise Exception('Invalid tracker type!')

    return tracker
