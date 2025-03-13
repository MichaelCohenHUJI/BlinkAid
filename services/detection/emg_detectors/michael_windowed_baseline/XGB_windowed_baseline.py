import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from typing import Optional
import math

from services.common.models.emg import EmgModel
from services.detection.emg_detectors.base_emg_detector import BaseEmgDetector
from services.common.enums.detection_types import DetectionType
from services.common.models.detection import DetectionModel



PROJ_DIR = '/home/michael/Desktop/BlinkAid/miniproject/'

class XGB_windowed_baseline(BaseEmgDetector):
    def __init__(self,
                 model_path=PROJ_DIR + "models/raz_xg_windowed_stdized_16pc_2025-03-12_20-53-06/raz_xg_windowed_stdized_16pc_2025-03-12_20-53-06.pkl",
                 sample_rate=250,
                 **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self._model_path = model_path
        self._model: xgb.XGBClassifier = joblib.load(self._model_path)
        self._meta_path = model_path[:-4] + '_metadata.pkl'
        self._meta = joblib.load(self._meta_path)
        self._buffer: list[EmgModel] = []
        self._window_length = self._meta['window_length']
        self._window_size = int(self._window_length * self.sample_rate)
        self._window_overlap = 0  # 0 - 1
        self._step_size = math.ceil((1 - self._window_overlap) * self._window_size)
        self._total_steps = 0
        self._scaler = joblib.load(PROJ_DIR + self._meta['scaler_path'])
        self._pca_model = joblib.load(PROJ_DIR + self._meta['pca_model_path'])
        self._pca_columns = [f'PC{i + 1}' for i in range(self._pca_model.n_components)]
        self._window_columns = [f"{col}_t{t}" for t in range(self._window_size) for col in self._pca_columns]
        self._classes = ['neutral', DetectionType.BLINK, DetectionType.GAZE_LEFT, DetectionType.GAZE_RIGHT,
                         DetectionType.GAZE_CENTER, DetectionType.GAZE_UP, DetectionType.GAZE_DOWN]
        self._data_cols = [f"channel_{i+1}" for i in range(16)]
        self._last_detection_time = None
        self._cooldown = 0.2
        self._last_pred = None

    def detect(self, emg_data: EmgModel) -> Optional[dict]:

        self._buffer.append(emg_data)

        if len(self._buffer) < self._window_size:
            return None
        else:
            data = pd.DataFrame([emg.data for emg in self._buffer], columns=self._data_cols)
            scaled_data = pd.DataFrame(self._scaler.transform(data), columns=self._data_cols)
            pca_data = pd.DataFrame(self._pca_model.transform(scaled_data), columns=self._pca_columns)
            window = pd.DataFrame(pca_data.values.flatten().reshape(1, -1), columns=self._window_columns)
            prob = self._model.predict(window)
            pred = int(prob)
            confidence = 1

            # self._buffer.pop(0)
            self._buffer = self._buffer[self._step_size:] # todo

            if pred != 0:
                if self._last_detection_time is not None:
                    if self._last_detection_time + pd.Timedelta(seconds=self._cooldown) > emg_data.timestamp\
                            and pred == self._last_pred:
                        return None
                detection_time = emg_data.timestamp
                self._last_detection_time = detection_time
                self._last_pred = pred
                start_time = detection_time - pd.Timedelta(seconds=self._window_length)
                end_time = detection_time
                type = self._classes[pred]
                metadata = {"confidence": float(confidence)}
                return DetectionModel(start_time=start_time,
                                      end_time=end_time,
                                      type=type,
                                      confidence=confidence,
                                      metadata=metadata)
            else:
                return None








