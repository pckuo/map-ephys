
'''
MAP Motion Tracking Schema
'''

import datajoint as dj
import numpy as np

from . import experiment, lab
from . import get_schema_name, create_schema_settings

schema = dj.schema(get_schema_name('tracking'), **create_schema_settings)
[experiment]  # NOQA flake8


@schema
class TrackingDevice(dj.Lookup):
    definition = """
    tracking_device:                    varchar(20)     # device type/function
    ---
    tracking_position:                  varchar(20)     # device position
    sampling_rate:                      decimal(8, 4)   # sampling rate (Hz)
    tracking_device_description:        varchar(100)    # device description
    """
    contents = [
        ('Camera 0', 'side', 1/0.0034, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
        ('Camera 1', 'bottom', 1/0.0034, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
        ('Camera 2', 'body', 1/0.01, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
        ('Camera 3', 'side', 1 / 0.0034, 'Blackfly S BFS-U3-04S2M-CS (FLIR)'),
        ('Camera 4', 'bottom', 1 / 0.0034, 'Blackfly S BFS-U3-04S2M-CS (FLIR)'),
        ('Camera 5', 'body', 1 / 0.01, 'Blackfly S BFS-U3-04S2M-CS (FLIR)'),
    ]


@schema
class Tracking(dj.Imported):
    '''
    Video feature tracking.
    Position values in px; camera location is fixed & real-world position
    can be computed from px values.
    '''

    definition = """
    -> experiment.SessionTrial
    -> TrackingDevice
    ---
    tracking_samples: int             # number of events (possibly frame number, relative to the start of the trial)
    """
    
    class Frame(dj.Part):
        definition = """
        -> Tracking
        ---
        frame_time: longblob   # Global session-wise time (in sec)
        """

    class NoseTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        nose_x:                 longblob        # nose x location (px)
        nose_y:                 longblob        # nose y location (px)
        nose_likelihood:        longblob        # nose location likelihood
        """

    class TongueTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        tongue_x:               longblob        # tongue x location (px)
        tongue_y:               longblob        # tongue y location (px)
        tongue_likelihood:      longblob        # tongue location likelihood
        """
        
    class TongueSideTracking(dj.Part):
        definition = """
        -> Tracking
        side:               varchar(36)     # leftfront, rightfront, leftback, rightback, ...
        ---
        tongue_side_x:               longblob        # tongue x location (px)
        tongue_side_y:               longblob        # tongue y location (px)
        tongue_side_likelihood:      longblob        # tongue location likelihood
        """

    class JawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        jaw_x:                  longblob        # jaw x location (px)
        jaw_y:                  longblob        # jaw y location (px)
        jaw_likelihood:         longblob        # jaw location likelihood
        """

    class LeftPawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        left_paw_x:             longblob        # left paw x location (px)
        left_paw_y:             longblob        # left paw y location (px)
        left_paw_likelihood:    longblob        # left paw location likelihood
        """

    class RightPawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        right_paw_x:            longblob        # right paw x location (px)
        right_paw_y:            longblob        # right_paw y location (px)
        right_paw_likelihood:   longblob        # right_paw location likelihood
        """

    class LickPortTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        lickport_x:            longblob        # right paw x location (px)
        lickport_y:            longblob        # right_paw y location (px)
        lickport_likelihood:   longblob        # right_paw location likelihood
        """

    class WhiskerTracking(dj.Part):
        definition = """
        -> Tracking
        whisker_name:         varchar(36)
        ---
        whisker_x:            longblob        # whisker x location (px)
        whisker_y:            longblob        # whisker y location (px)
        whisker_likelihood:   longblob        # whisker location likelihood
        """

    @property
    def tracking_features(self):
        return {'NoseTracking': Tracking.NoseTracking,
                'TongueTracking': Tracking.TongueTracking,
                'JawTracking': Tracking.JawTracking,
                'LeftPawTracking': Tracking.LeftPawTracking,
                'RightPawTracking': Tracking.RightPawTracking,
                'LickPortTracking': Tracking.LickPortTracking,
                'WhiskerTracking': Tracking.WhiskerTracking,
                
                # For foraging tracking
                'nose': Tracking.NoseTracking,
                'tongue': Tracking.TongueTracking,
                'tongue_side': Tracking.TongueSideTracking,
                'jaw': Tracking.JawTracking,
                'left_paw': Tracking.LeftPawTracking,
                'right_paw': Tracking.RightPawTracking,
                'whisker': Tracking.WhiskerTracking,                
                }


@schema
class TrackedWhisker(dj.Manual):
    definition = """
    -> Tracking.WhiskerTracking
    """

    class Whisker(dj.Part):
        definition = """
        -> master
        -> lab.Whisker
        """


# ------------------------ Quality Control Metrics -----------------------


@schema
class TrackingQC(dj.Computed):
    definition = """
    -> Tracking
    tracked_feature: varchar(32)  # e.g. RightPaw, LickPort
    ---
    bad_percentage: float  # percentage of bad frames out of all frames 
    bad_frames: longblob  # array of frame indices that are "bad"
    """

    threshold_mapper = {('RRig2', 'side', 'NoseTracking'): 20,
                        ('RRig2', 'side', 'JawTracking'): 20,
                        ('RRig-MTL', 'side', 'JawTracking'): 20,
                        ('RRig-MTL', 'bottom', 'JawTracking'): 20}

    def make(self, key):
        rig = (experiment.Session & key).fetch1('rig')
        device, device_position = (TrackingDevice & key).fetch1('tracking_device', 'tracking_position')

        tracking_qc_list = []
        for feature_name, feature_table in Tracking().tracking_features.items():
            if feature_name in ('JawTracking', 'NoseTracking'):
                if not feature_table & key:
                    continue

                bad_threshold = self.threshold_mapper[(rig, device_position, feature_name)]
                tracking_data = (Tracking * feature_table & key).fetch1()

                attribute_prefix = feature_name.replace('Tracking', '').lower()

                x_diff = np.diff(tracking_data[attribute_prefix + '_x'])
                y_diff = np.diff(tracking_data[attribute_prefix + '_y'])
                bad_frames = np.where(np.logical_or(x_diff > bad_threshold, y_diff > bad_threshold))[0]
                bad_percentage = len(bad_frames) / tracking_data['tracking_samples'] * 100

                tracking_qc_list.append({**key, 'tracked_feature': feature_name,
                                         'bad_percentage': bad_percentage,
                                         'bad_frames': bad_frames})

        self.insert(tracking_qc_list)

