import joblib
from xgboost import XGBClassifier
import logging
import pandas as pd
import xgboost as xgb
from typing import Optional
import math
from services.common.models.emg import EmgModel
from services.detection.emg_detectors.base_emg_detector import BaseEmgDetector
from services.common.models.detection import DetectionModel
from pca_ica_exploration import train_pca, apply_train_pca
from tqdm import tqdm
from windowing import create_windows
from training_helpers import collect_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)


class BlinkAidXGB(BaseEmgDetector):
    def __init__(self,
                 classes,
                 sample_rate=250,
                 training_window_overlap=0.99,  # 0 - 1, for training and validation
                 inf_window_overlap=0,  # 0 - 1, for inference data only
                 window_length=0.3,  # seconds
                 cooldown=0.4,  # seconds, cooldown time between 2 identical predictions
                 num_channels=16,
                 p_components=3,
                 split_ratio=0.2,  # 0 - 1, fraction of validation set out of the input data
                 **kwargs):
        """
        Initialize XGBoost model for BlinkAid datasets
        :param classes: list of string representations of the classes
        :param model_path:
        :param sample_rate: data sample rate
        :param training_window_overlap: 0 - 1, consecutive windows overlap, for training and validation
        :param inf_window_overlap: 0 - 1, consecutive windows overlap, for inference data only
        :param window_length: seconds
        :param cooldown: seconds, cooldown time between 2 identical predictions
        :param num_channels: number of channels in the sampled data
        :param p_components: number of pca components to train with
        :param split_ratio: 0 - 1, fraction of validation set out of the input data
        :param kwargs:
        """
        super().__init__(**kwargs)

        self._scaler = None
        self._pca_model = None

        # initialize needed params
        self._model: XGBClassifier = None
        self._fitted = False
        self._n_classes = len(classes)
        self._classes = classes
        self._classes_strings = []
        for i, c in enumerate(self._classes):
            self._classes_strings.append(c.value + f' ({i})')
        self._window_length = window_length
        self._split_ratio = split_ratio
        self._cooldown = cooldown
        self._p_components = p_components
        self._window_size = int(self._window_length * sample_rate)
        self._training_window_overlap = training_window_overlap
        self._inference_step_size = math.ceil((1 - inf_window_overlap) * self._window_size)
        self._last_detection_time = None
        self._last_pred = None
        self._buffer: list[EmgModel] = []
        self._confusion_matrix = None  # last training confusion matrix
        self._validation_report = None  # last validation report
        self._validation_report_dict = None  # last validation report dictionary
        self._accuracy_score = None  # last accuracy score

        # initialize data columns & class names
        self._data_cols = [f"channel_{i + 1}" for i in range(num_channels)]
        self._pca_columns = [f'PC{i + 1}' for i in range(self._p_components)]
        self._window_columns = [f"{col}_t{t}" for t in range(self._window_size) for col in self._pca_columns]

        logger.info(f"🔍 Model loaded successfully.")

    def is_fitted(self) -> bool:
        return self._fitted


    def fit(self, data_paths_dict, subj_list):
        """
        main framework for training.
        Stages:
        1. collect data (train + test)
        2. trains standardization and pca models on train set and applies them on all data
        3. creates sliding windows for training and test sets
        4. training xgboost model on train set, and returning a performance report on test set
        5. saving the models mentioned above, the performance report and creates a tensorboard report for the trained model
        """
        """Stage 1"""
        # collect data from all files
        train_dfs, val_dfs = collect_data(data_paths_dict, subj_list, self._split_ratio)

        """Stage 2"""
        # train standardization and pca models on the train data
        df_all_train = pd.concat(train_dfs, ignore_index=True)
        df_all_train_pca, pca_results, pca, scaler = train_pca(df_all_train, self._p_components)
        self._scaler = scaler
        self._pca_model = pca
        # apply pca to whole data
        train_dfs_pca = [apply_train_pca(df, scaler, pca) for df in train_dfs]
        test_dfs_pca = [apply_train_pca(df, scaler, pca) for df in val_dfs]

        """Stage 3"""
        # create labeled windows from annotated samples
        train_windows = []
        test_windows = []
        for df in tqdm(train_dfs_pca):
            windows = create_windows(df, self._window_length, self._training_window_overlap)
            train_windows.append(windows)
        for df in tqdm(test_dfs_pca):
            windows = create_windows(df, self._window_length, self._training_window_overlap)
            test_windows.append(windows)
        train_windows_df = pd.concat(train_windows, ignore_index=True)
        test_windows_df = pd.concat(test_windows, ignore_index=True)

        """Stage 4"""
        # train model
        # shuffle train & validation sets
        traindf = train_windows_df.sample(frac=1).reset_index(drop=True)
        testdf = test_windows_df.sample(frac=1).reset_index(drop=True)
        # Separate features and labels
        X_train = traindf.drop(columns=['timestamp', 'label'])
        y_train = traindf['label']
        X_test = testdf.drop(columns=['timestamp', 'label'])
        y_test = testdf['label']

        # Train XGBoost model
        model = xgb.XGBClassifier(
            eval_metric='mlogloss',  # Multi-class log loss
            objective='multi:softprob',  # Softmax output
            num_class=self._n_classes,  # Replace N with the number of classes
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self._accuracy_score = accuracy
        print(f"Model Accuracy: {accuracy:.4f}")

        # Compute confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        self._confusion_matrix = cm
        report = classification_report(y_test, y_pred, target_names=self._classes_strings)
        self._validation_report = report
        report_dict = classification_report(y_test, y_pred, target_names=self._classes_strings, output_dict=True)
        self._validation_report_dict = report_dict
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

        self._model = model
        self._fitted = True
        return self


    def get_performance_report(self):
        if not self._fitted:
            raise ValueError("Model was not fitted yet.")
        return self._accuracy_score, self._confusion_matrix, self._validation_report, self._validation_report_dict

    def continue_fit(self, data_paths_dict, subj_list):  # todo test
        """

        :param data_paths_dict:
        :param subj_list:
        :return:
        """
        if not self._fitted:
            raise ValueError("Cannot continue training. The model hasn't been trained yet.")

        """Stage 1"""
        # collect data from all files
        train_dfs, val_dfs = collect_data(data_paths_dict, subj_list, self._split_ratio)

        """Stage 2"""
        # apply pca to whole data
        train_dfs_pca = [apply_train_pca(df, self._scaler, self._pca_model) for df in train_dfs]
        test_dfs_pca = [apply_train_pca(df, self._scaler, self._pca_model) for df in val_dfs]

        """Stage 3"""
        # create labeled windows from annotated samples
        train_windows = []
        test_windows = []
        for df in tqdm(train_dfs_pca):
            windows = create_windows(df, self._window_length, self._training_window_overlap)
            train_windows.append(windows)
        for df in tqdm(test_dfs_pca):
            windows = create_windows(df, self._window_length, self._training_window_overlap)
            test_windows.append(windows)
        train_windows_df = pd.concat(train_windows, ignore_index=True)
        test_windows_df = pd.concat(test_windows, ignore_index=True)

        """Stage 4"""
        # shuffle train & validation sets
        traindf = train_windows_df.sample(frac=1).reset_index(drop=True)
        testdf = test_windows_df.sample(frac=1).reset_index(drop=True)
        # Separate features and labels
        X_train = traindf.drop(columns=['timestamp', 'label'])
        y_train = traindf['label']
        X_test = testdf.drop(columns=['timestamp', 'label'])
        y_test = testdf['label']

        # train model
        self._model.fit(X_train, y_train, xgb_model=self._model.get_booster())

        # Predict on test set
        y_pred = self._model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self._accuracy_score = accuracy
        print(f"Model Accuracy: {accuracy:.4f}")

        # Compute confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        self._confusion_matrix = cm
        report = classification_report(y_test, y_pred, target_names=self._classes_strings)
        self._validation_report = report
        report_dict = classification_report(y_test, y_pred, target_names=self._classes_strings, output_dict=True)
        self._validation_report_dict = report_dict
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)
        return self

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
            self._buffer = self._buffer[self._inference_step_size:]  # todo talk to raz about step size and overlap

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

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)

