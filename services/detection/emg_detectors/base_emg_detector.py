# File: services/detection/emg_detectors/base_emg_detector.py
from services.common.models.detection import DetectionModel
from services.common.models.emg import EmgModel

from abc import ABC, abstractmethod
from typing import Optional


class BaseEmgDetector(ABC):
    @abstractmethod
    def detect(self, emg_model: EmgModel) -> Optional[DetectionModel]:
        raise NotImplementedError("Detector must implement the '_detect' method.")
