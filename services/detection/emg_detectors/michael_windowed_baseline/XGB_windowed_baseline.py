import numpy as np
import pandas as pd
import xgboost as xgb
from services.detection.emg_detectors.base_emg_detector import BaseEmgDetector
import joblib
from services.common.models.emg import EmgModel
from typing import Optional
from windowing import create_windows



class XGB_windowed_baseline(BaseEmgDetector):
    def __init__(self,
                 model_path="models/raz_xg_windowed_stdized_16pc_2025-03-12_16-25-45/raz_xg_windowed_stdized_16pc_2025-03-12_16-25-45.pkl",
                 sample_rate=250,
                 **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self._model_path = model_path
        self._model = joblib.load(self._model_path)
        self._meta_path = model_path[:-4] + '_metadata.pkl'
        self._meta = joblib.load(self._meta_path)
        self._buffer: list[EmgModel] = []
        self._window_length = self._meta['window_length']
        self._window_size = self._window_length * self.sample_rate
        self._window_overlap = self._meta['overlap']  # 0 - 1
        self._step_size = (1 - self._window_overlap) * self._window_size
        self._total_steps = 0
        self._scaler = joblib.load(self._meta['scaler_path'])
        self._pca_model = joblib.load(self._meta['pca_model_path'])
        self._pca_columns = [f'PC{i + 1}' for i in range(self._pca_model.n_components)]
        self._window_columns = [f"{col}_t{t}" for t in range(self._window_size) for col in self._pca_columns]


    def detect(self, emg_data: EmgModel) -> Optional[dict]:
        self._total_steps += 1
        self._buffer.append(emg_data)

        if len(self._buffer) < self._window_size:
            return None
        elif self._total_steps % self._step_size == 0:
            data = pd.DataFrame([emg.data for emg in self._buffer])
            scaled_data = pd.DataFrame(self._scaler.transform(data))
            pca_data = pd.DataFrame(self._pca_model.transform(scaled_data), columns=self._pca_columns)
            window = data.values.flatten()





