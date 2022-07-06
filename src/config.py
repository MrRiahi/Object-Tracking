

class Config:
    """
    Configuration of the tracking algorithms
    """

    SAVE_RESULT = True
    # Options:
    # MILTracker, KCFTracker, CSRTTracker, MOSSETacker, TLDTracker, BoostingTracker,
    # DaSiamRPNTacker, GOTURNTacker
    TRACKER_TYPE = 'CSRTTracker'
