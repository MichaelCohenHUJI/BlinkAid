# File: services/common/enums/detection_types.py
from enum import Enum, auto


class DetectionType(Enum):
    BLINK = "Blink"
    GAZE_LEFT = "Gaze Left"
    GAZE_RIGHT = "Gaze Right"
    GAZE_CENTER = "Gaze Center"
    GAZE_UP = "Gaze Up"
    GAZE_DOWN = "Gaze Down"
    NOISE = "Noise"
