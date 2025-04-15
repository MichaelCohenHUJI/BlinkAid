from services.common.enums.detection_types import DetectionType

CLASSES_TO_LABELS = {
    DetectionType.NEUTRAL: 0,
    DetectionType.BLINK: 1,
    DetectionType.GAZE_LEFT: 2,
    DetectionType.GAZE_RIGHT: 3,
    DetectionType.GAZE_UP: 4,
    DetectionType.GAZE_DOWN: 5,
    DetectionType.NOISE: 6,
}

LABELS_TO_CLASSES= {
    v: k for k, v in CLASSES_TO_LABELS.items()
}
