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
from tqdm import tqdm
from services.detection.emg_detectors.michael_windowed_baseline.training_helpers import collect_data, train_pca, apply_train_pca, create_datasets
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

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
                 xgb_params=None,
                 **kwargs):
        """
        Initialize XGBoost model for BlinkAid datasets
        :param classes: list of string representations of the classes
        :param sample_rate: data sample rate
        :param training_window_overlap: 0 - 1, consecutive windows overlap, for training and validation
        :param inf_window_overlap: 0 - 1, consecutive windows overlap, for inference data only
        :param window_length: seconds
        :param cooldown: seconds, cooldown time between 2 identical predictions
        :param num_channels: number of channels in the sampled data
        :param p_components: number of pca components to train with
        :param split_ratio: 0 - 1, fraction of validation set out of the input data
        :param xgb_params: parameters for xgboost model
        :param kwargs:
        """
        super().__init__(**kwargs)

        self._xgb_params = xgb_params or {}
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
        self._sample_rate = sample_rate
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

    def get_scaler(self):
        return self._scaler

    def get_pca_model(self):
        return self._pca_model

    def get_performance_report(self):
        if not self._fitted:
            raise ValueError("Model was not fitted yet.")
        return self._accuracy_score, self._confusion_matrix, self._validation_report, self._validation_report_dict

    def fit(self, data_paths_dict, subj_list):
        """
        main framework for training.
        Stages:
        1. collect data (train + test)
        2. trains standardization and pca models on train set and applies them on all data
        3. creates sliding windows for training and test sets
        4. training xgboost model on train set, and returning a performance report on test set
        5. saving the models mentioned above and the performance report
        :param data_paths_dict: dictionary with data paths, keys are the subjects the data was recorded on
        :param subj_list: the list of subjects to train on
        :return: self
        """
        """Stage 1"""
        # collect data from all files
        train_dfs, val_dfs = collect_data(data_paths_dict, subj_list, self._split_ratio)

        """Stage 2"""
        # train standardization and pca/ica models on the train data
        df_all_train = pd.concat(train_dfs, ignore_index=True)
        df_all_train_pca, pca, scaler = train_pca(df_all_train, self._p_components)
        self._scaler = scaler
        self._pca_model = pca
        # apply pca to whole data
        train_dfs_processed = [apply_train_pca(df, scaler, pca) for df in train_dfs]
        val_dfs_processed = [apply_train_pca(df, scaler, pca) for df in val_dfs]

        """Stage 3"""
        # create labeled windows from annotated samples
        X_train, X_test, y_train, y_test = create_datasets(train_dfs_processed, val_dfs_processed, self._window_length,
                                                           self._training_window_overlap, self._sample_rate)
        """Stage 4"""
        # Train XGBoost model
        model = xgb.XGBClassifier(
            eval_metric='mlogloss',  # Multi-class log loss
            objective='multi:softprob',  # Softmax output
            num_class=self._n_classes,  # Replace N with the number of classes
            n_jobs=-1,
            **self._xgb_params
        )
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Compute confusion matrix and classification report
        self.calculate_model_performance(y_pred, y_test)

        self._model = model
        self._fitted = True
        return self

    def calculate_model_performance(self, y_pred, y_test):
        """
        calculate performance report on test set, saves and prints it
        :param y_pred: predicted labels
        :param y_test: real labels
        :return:
        """
        # Compute confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        self._confusion_matrix = cm
        # Ensure label names match
        present_labels = sorted(set(np.unique(y_test)).union(np.unique(y_pred)))
        report = classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            target_names=[self._classes_strings[i] for i in present_labels]
        )
        self._validation_report = report
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            target_names=[self._classes_strings[i] for i in present_labels],
            output_dict=True
        )
        self._validation_report_dict = report_dict
        self._accuracy_score = report_dict['accuracy']
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

    def fit_hpt(self, X_train, y_train, X_test, y_test):
        """
        helper model fitting method to use only for hyperparameter tuning
        :param X_train: training design matrix
        :param y_train: training response vector
        :param X_test: test design matrix
        :param y_test: test response vector
        :return:
        """
        if self._fitted:
            raise ValueError("Model already fitted")
        # Train XGBoost model
        model = xgb.XGBClassifier(
            eval_metric='mlogloss',  # Multi-class log loss
            objective='multi:softprob',  # Softmax output
            num_class=self._n_classes,
            n_jobs=-1,
            **self._xgb_params
        )
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # calculate performance report
        present_labels = sorted(set(np.unique(y_test)).union(np.unique(y_pred)))
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            target_names=[self._classes_strings[i] for i in present_labels],
            output_dict=True
        )
        self._validation_report_dict = report_dict
        self._accuracy_score = report_dict['accuracy']
        self._model = model
        self._fitted = True
        return self

    def continue_fit(self, data_paths_dict, subj_list):  # todo test method
        """
        continue training an existing model on new data
        :param data_paths_dict: dictionary with data paths, keys are the subjects the data was recorded on
        :param subj_list: the list of subjects to train on
        :return:
        """
        if not self._fitted:
            raise ValueError("Cannot continue training. The model hasn't been trained yet.")

        # collect data from all files
        train_dfs, val_dfs = collect_data(data_paths_dict, subj_list, self._split_ratio)

        # apply pca to whole data
        train_dfs_processed = [apply_train_pca(df, self._scaler, self._pca_model) for df in train_dfs]
        val_dfs_processed = [apply_train_pca(df, self._scaler, self._pca_model) for df in val_dfs]

        # create labeled windows from annotated samples
        X_train, X_test, y_train, y_test = create_datasets(train_dfs_processed, val_dfs_processed,
                                                                self._window_length, self._training_window_overlap,
                                                                self._sample_rate)

        # train model
        self._model.fit(X_train, y_train, xgb_model=self._model.get_booster())

        # Predict on test set
        y_pred = self._model.predict(X_test)

        # Compute confusion matrix and classification report
        self.calculate_model_performance(y_pred, y_test)

        return self

    def test_model(self, annotated_data_paths, subj_list):
        """
        a method used for model testing after training and validation.
        prints the results of the model on test set.
        :param annotated_data_paths: test data paths
        :param subj_list: subjects to test on
        :return:
        """
        if not self._fitted:
            raise ValueError("Model was not fitted yet.")
        # Collect annotated input data
        _, test_dfs = collect_data(annotated_data_paths, subj_list, 0.2)
        # Apply PCA transformation to each annotated dataset
        test_dfs_processed = [apply_train_pca(df, self._scaler, self._pca_model) for df in test_dfs]
        # Create windows from the PCA-transformed data
        X_test, _, y_test, __ = create_datasets(test_dfs_processed, [], self._window_length,
                                                self._training_window_overlap, self._sample_rate)
        # Get predictions from the existing model
        y_pred = self._model.predict(X_test)

        # Compute confusion matrix and classification report
        self.calculate_model_performance(y_pred, y_test)
        return self._validation_report

    def detect(self, emg_data: EmgModel) -> Optional[dict]:

        self._buffer.append(emg_data)

        if len(self._buffer) < self._window_size:
            return None
        else:
            data = pd.DataFrame([emg.data for emg in self._buffer], columns=self._data_cols)
            scaled_data = pd.DataFrame(self._scaler.transform(data), columns=self._data_cols)
            pca_data = pd.DataFrame(self._pca_model.transform(scaled_data), columns=self._pca_columns)
            window = pd.DataFrame(pca_data.values.flatten().reshape(1, -1), columns=self._window_columns)
            probs = self._model.predict_proba(window)[0]
            pred = np.argmax(probs)
            confidence = probs[pred]

            # self._buffer.pop(0)
            self._buffer = self._buffer[self._inference_step_size:]

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

