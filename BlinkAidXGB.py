from xgboost import XGBClassifier

class BlinkAidXGB(XGBClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._window_length = None  # seconds
        self._window_step = None  # seconds
        self._num_classes = None
        self._pca_model_path = None
        self._scaler_path = None


    def set_window_length(self, value):
        self._window_length = value

    def get_window_length(self):
        return self._window_length

    def set_window_step(self, value):
        self._window_step = value

    def get_window_step(self):
        return self._window_step

    def set_num_classes(self, value):
        self._num_classes = value

    def get_num_classes(self):
        return self._num_classes

    def set_pca_model_path(self, value):
        self._pca_model_path = value

    def get_pca_model_path(self):
        return self._pca_model_path

    def set_scaler_path(self, value):
        self._scaler_path = value

    def get_scaler_path(self):
        return self._scaler_path

