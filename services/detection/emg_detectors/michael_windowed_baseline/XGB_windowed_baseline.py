import logging
import pandas as pd
import xgboost as xgb
import joblib
from typing import Optional
import math
from services.common.models.emg import EmgModel
from services.detection.emg_detectors.base_emg_detector import BaseEmgDetector
from services.common.enums.detection_types import DetectionType
from services.common.models.detection import DetectionModel
from services.detection.emg_detectors.michael_windowed_baseline import MICHAEL_DETECTOR_DIR

logger = logging.getLogger(__name__)


class XGB_windowed_baseline(BaseEmgDetector):
    def __init__(self,
                 model_path=str(MICHAEL_DETECTOR_DIR) + "/models/raz_xg_windowed_stdized_16pc_2025-03-12_20-53-06/raz_xg_windowed_stdized_16pc_2025-03-12_20-53-06.pkl",
                 sample_rate=250,
                 window_overlap=0,  # 0 - 1, for inference data only
                 cooldown=0.2,  # cooldown time between 2 identical predictions
                 num_channels=16,  # same as in training
                 **kwargs):
        logger.info(f"🔍 Loading model from {model_path}...")
        super().__init__(**kwargs)

        # load models & meta data
        self._model_path = model_path
        self._meta_path = model_path[:-4] + '_metadata.pkl'
        self._model: xgb.XGBClassifier = joblib.load(self._model_path)
        self._meta = joblib.load(self._meta_path)
        self._scaler = joblib.load(str(MICHAEL_DETECTOR_DIR) + "/" + self._meta['scaler_path'])
        self._pca_model = joblib.load(str(MICHAEL_DETECTOR_DIR) + "/" + self._meta['pca_model_path'])

        # initialize needed params
        self._window_length = self._meta['window_length']
        self._window_size = int(self._window_length * sample_rate)
        self._step_size = math.ceil((1 - window_overlap) * self._window_size)
        self._last_detection_time = None
        self._cooldown = cooldown
        self._last_pred = None
        self._buffer: list[EmgModel] = []

        # initialize data columns & class names
        self._data_cols = [f"channel_{i + 1}" for i in range(num_channels)]
        self._pca_columns = [f'PC{i + 1}' for i in range(self._pca_model.n_components)]
        self._window_columns = [f"{col}_t{t}" for t in range(self._window_size) for col in self._pca_columns]
        self._classes = ['neutral', DetectionType.BLINK, DetectionType.GAZE_LEFT, DetectionType.GAZE_RIGHT,
                         DetectionType.GAZE_CENTER, DetectionType.GAZE_UP, DetectionType.GAZE_DOWN]

        logger.info(f"🔍 Model loaded successfully.")


    def fit(self):
        pass

    def detect(self, emg_data: EmgModel) -> Optional[dict]:

        self._buffer.append(emg_data)

        if len(self._buffer) < self._window_size:
            return None
        else:
            data = pd.DataFrame([emg.data for emg in self._buffer], columns=self._data_cols)
            scaled_data = pd.DataFrame(self._scaler.transform(data), columns=self._data_cols)
            pca_data = pd.DataFrame(self._pca_model.transform(scaled_data), columns=self._pca_columns)
            window = pd.DataFrame(pca_data.values.flatten().reshape(1, -1), columns=self._window_columns)
            pred = self._model.predict(window)[0]
            confidence = self._model.predict_proba(window)[0][pred]

            # self._buffer.pop(0)
            self._buffer = self._buffer[self._step_size:]  # todo talk to raz about step size and overlap

            if pred != 0:
                if self._last_detection_time is not None:  # make sure we don't classify single event as two in a row
                    if self._last_detection_time + pd.Timedelta(seconds=self._cooldown) > emg_data.timestamp \
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
