# File: services/detection/detection_service.py
import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from services.common.enums.detector_types import DetectorTypes
from services.common.models.detection import DetectionModel
from services.common.models.emg import EmgModel
from services.common.pubsub import PubSub
from services.detection.emg_detectors.base_emg_detector import BaseEmgDetector
from services.detection.emg_detectors.examples.blink_detector_cnn import BlinkDetectorCNN
from services.detection.emg_detectors.examples.blink_detector_threshold_voting import BlinkDetectorThresholdVoting

logger = logging.getLogger(__name__)


@dataclass
class DetectorEntry:
    detector: BaseEmgDetector
    is_enabled: bool = True
    last_detection: Optional[DetectionModel] = None
    allow_overlaps: bool = False
    detections_count: int = 0

    def status(self):
        return {
            "is_enabled": self.is_enabled,
            "last_detection": self.last_detection,
            "allow_overlaps": self.allow_overlaps,
            "detections_count": self.detections_count,
        }


class DetectionService:
    DEFAULT_DETECTORS_TABLE = {
        DetectorTypes.BLINK_DETECTOR_THRESHOLD_VOTING:
            DetectorEntry(
                detector=BlinkDetectorThresholdVoting(),
                # is_enabled=False
            ),
        DetectorTypes.BLINK_DETECTOR_CNN:
            DetectorEntry(
                detector=BlinkDetectorCNN(),
                is_enabled=False,
            ),
    }

    def __init__(self, pubsub: PubSub, num_channels: int):
        self._pubsub = pubsub
        self._num_channels = num_channels
        self._detectors_table: dict[str: DetectorEntry] = deepcopy(self.DEFAULT_DETECTORS_TABLE)

    async def start(self):
        enabled_detector_names = [name for name, entry in self._detectors_table.items() if entry.is_enabled]
        logger.info(f"🔍 Starting EMG detection service with: {enabled_detector_names}")
        self._detection_task = asyncio.create_task(self._pubsub.subscribe(
            channel=PubSub.Channels.EMG,
            message_handler=self._run_enabled_detectors,
            message_class=EmgModel))
        logger.debug("🔍 EMG detection service started")

    async def stop(self):
        logger.info("🛑 Stopping EMG detection service")
        self._detection_task.cancel()
        logger.debug("🛑 EMG detection service stopped")

    async def publish_detection(self, detector_name: str, detection: DetectionModel):
        logger.info(f"{detector_name} detected: {detection}")
        await self._pubsub.publish(PubSub.Channels.DETECTIONS, detection)

    def enable_detector(self, detector_name: DetectorTypes):
        if detector_name not in self._detectors_table:
            raise KeyError(f"Detector {detector_name} not found.")
        self._detectors_table[detector_name].is_enabled = True

    def disable_detector(self, detector_name: str):
        if detector_name not in self._detectors_table:
            raise KeyError(f"Detector {detector_name} not found.")
        self._detectors_table[detector_name].is_enabled = False

    def status(self):
        return {detector_name : detector_entry.status() for detector_name, detector_entry in self._detectors_table.items()}

    async def _run_enabled_detectors(self, emg_sample: EmgModel):
        for detector_name, detector_entry in self._detectors_table.items():
            if detector_entry.is_enabled:
                try:
                    detection = await self._run_detector(detector_entry, emg_sample)
                    if detection:
                        await self.publish_detection(detector_name, detection)
                except Exception as e:
                    logger.error(f"❌ Error running detector {detector_name}: {e}")
                    continue

    async def _run_detector(self, detector_entry: DetectorEntry, emg_sample: EmgModel) -> Optional[DetectionModel]:
        try:
            detection = await asyncio.to_thread(detector_entry.detector.detect, emg_sample)
            if detection:
                if detector_entry.allow_overlaps or not (detection.overlaps(detector_entry.last_detection)):
                    detector_entry.detections_count += 1
                    detector_entry.last_detection = detection
                    return detection
        except Exception as e:
            logger.error(f"❌ Error running detector {detector_entry.detector}: {e}")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
