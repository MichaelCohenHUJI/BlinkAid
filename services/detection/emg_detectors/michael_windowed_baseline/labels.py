from services.common.enums.detection_types import DetectionType

LABELS_TO_CLASSES = {
    DetectionType.NEUTRAL: 0,
    DetectionType.BLINK: 1,
    DetectionType.GAZE_LEFT: 2,
    DetectionType.GAZE_RIGHT: 3,
    DetectionType.GAZE_CENTER: 4,
    DetectionType.GAZE_UP: 5,
    DetectionType.GAZE_DOWN: 6,
    DetectionType.NOISE: 7,
}

CLASSES_TO_LABELS = {
    v: k for k, v in LABELS_TO_CLASSES.items()
}
