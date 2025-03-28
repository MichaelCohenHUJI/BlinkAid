from xgboost import XGBClassifier
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
from datetime import datetime
from pca_ica_exploration import train_pca, apply_train_pca
from tqdm import tqdm
from windowing import create_windows
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import io
import torch
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)


class BlinkAidXGB(BaseEmgDetector):
    def __init__(self,
                 model_path=str(MICHAEL_DETECTOR_DIR) + "/models/raz_xg_windowed_stdized_16pc_2025-03-12_20-53-06/raz_xg_windowed_stdized_16pc_2025-03-12_20-53-06.pkl",
                 sample_rate=250,
                 n_classes=7,
                 training_window_overlap=0.99,  # 0 - 1, for training and validation
                 inf_window_overlap=0,  # 0 - 1, for inference data only
                 window_length=0.3,  # seconds
                 cooldown=0.2,  # cooldown time between 2 identical predictions
                 num_channels=16,  # same as in training
                 p_components=3,
                 split_ratio=0.2,  # 0 - 1, fraction of validation set out of the input data
                 **kwargs):
        logger.info(f"🔍 Loading model from {model_path}...")
        super().__init__(**kwargs)

        # save model path and metadata
        self._model_path = None
        self._meta_path = None
        # self._model: xgb.XGBClassifier = joblib.load(self._model_path)
        # self._meta = joblib.load(self._meta_path)
        self._meta = None
        # self._scaler = joblib.load(str(MICHAEL_DETECTOR_DIR) + "/" + self._meta['scaler_path'])
        self._scaler = None
        # self._pca_model = joblib.load(str(MICHAEL_DETECTOR_DIR) + "/" + self._meta['pca_model_path'])
        self._pca_model = None

        # initialize needed params
        self._n_classes = n_classes
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

        # initialize data columns & class names
        self._data_cols = [f"channel_{i + 1}" for i in range(num_channels)]
        self._pca_columns = [f'PC{i + 1}' for i in range(self._p_components)]
        self._window_columns = [f"{col}_t{t}" for t in range(self._window_size) for col in self._pca_columns]
        self._classes = ['neutral', DetectionType.BLINK, DetectionType.GAZE_LEFT, DetectionType.GAZE_RIGHT,
                         DetectionType.GAZE_CENTER, DetectionType.GAZE_UP, DetectionType.GAZE_DOWN]

        logger.info(f"🔍 Model loaded successfully.")


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
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        """Stage 1"""
        # collect data from all files
        from training_helpers import collect_data
        train_dfs, val_dfs = collect_data(data_paths_dict, subj_list, self._split_ratio)
        trained_on = ''
        for subj in subj_list:
            trained_on += subj + '_'

        """Stage 2"""
        # train standardization and pca models on the train data
        df_all_train = pd.concat(train_dfs, ignore_index=True)
        df_all_train_pca, pca_results, pca, scaler = train_pca(df_all_train, self._p_components)
        # apply pca to whole data
        train_dfs_pca = [apply_train_pca(df, pca, scaler) for df in train_dfs]
        test_dfs_pca = [apply_train_pca(df, pca, scaler) for df in val_dfs]

        """Stage 3"""
        # create labeled windows from annotated samples
        # window_length = 0.3  # seconds
        # overlap = 0.99  # 0 - 1
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

        """Stage 4"""   # todo continue refactoring
        # train model
        existing_model = 0
        n_classes = 7
        classes_strings = ['Neutral (0)', 'Blink (1)', 'Gaze Left (2)', 'Gaze Right (3)', 'Gaze Center (4)',
                           'Gaze Up (5)', 'Gaze Down (6)']
        # trained_model, cm, report, report_dict = train_xgb(train_windows_df, test_windows_df, n_classes,
        #                                                    classes_strings)
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
            num_class=n_classes,  # Replace N with the number of classes
        )
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")

        # Compute confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=classes_strings)
        report_dict = classification_report(y_test, y_pred, target_names=classes_strings, output_dict=True)
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

        """Stage 5"""
        # create model folder
        data_frac = str(int((1 - self._split_ratio) * 100)) + '%data_'
        model_name = trained_on + data_frac + "xg_windowed_stdized_" + str(self._p_components) + 'pc'
        model_folder = str(MICHAEL_DETECTOR_DIR) + "/models/" + model_name + "_" + timestamp + "/"
        os.makedirs(model_folder, exist_ok=True)

        # save models training report
        with open(model_folder + 'classification_report.txt', 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        # save pca and scaler data
        scaler_path = model_folder + model_name + "_" + timestamp + "_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        pca_model_path = model_folder + model_name + "_" + timestamp + "_pca_model.pkl"
        joblib.dump(pca, pca_model_path)

        # save model metadata
        model_meta = {}
        model_meta['scaler_path'] = scaler_path
        model_meta['p_components'] = p_components
        model_meta['pca_model_path'] = pca_model_path
        model_meta['window_length'] = window_length
        model_meta['overlap'] = overlap
        model_meta['n_classes'] = n_classes

        # save model
        model_path = model_folder + model_name + "_" + timestamp + ".pkl"
        joblib.dump(model, model_path)
        print(f"Trained model saved to {model_path}")

        meta_path = model_folder + model_name + "_" + timestamp + "_metadata.pkl"
        joblib.dump(model_meta, meta_path)

        # 🔥 Add Confusion Matrix Heatmap
        def plot_confusion_matrix(cm, labels):
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            image = torch.tensor(plt.imread(buf)).permute(2, 0, 1)[:3]  # [C, H, W]
            return image.unsqueeze(0)  # [1, C, H, W]

        # Initialize TensorBoard writer
        tb_log_dir = os.path.join(model_folder, 'tensorboard')
        writer = SummaryWriter(log_dir=tb_log_dir)

        # Log PCA explained variance to TensorBoard
        for i, var in enumerate(pca.explained_variance_ratio_):
            writer.add_scalar('PCA/Explained_Variance_Ratio_PC' + str(i + 1), var, 0)

        # Log classification metrics to TensorBoard
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                writer.add_scalar(f'Classification_Report/{label}_precision', metrics['precision'], 0)
                writer.add_scalar(f'Classification_Report/{label}_recall', metrics['recall'], 0)
                writer.add_scalar(f'Classification_Report/{label}_f1-score', metrics['f1-score'], 0)
            else:
                writer.add_scalar('Classification_Report/accuracy', report_dict['accuracy'], 0)

        cm_image = plot_confusion_matrix(cm, labels=[str(i) for i in range(n_classes)])
        writer.add_image('Confusion_Matrix', cm_image[0], 0)

        # Close TensorBoard writer
        writer.close()

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
