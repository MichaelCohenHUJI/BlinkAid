from services.detection.emg_detectors.michael_windowed_baseline.BlinkAidXGB import BlinkAidXGB
from datetime import datetime
from annotated_data_paths import DATA
from services.common.enums.detection_types import DetectionType
from services.detection.emg_detectors.michael_windowed_baseline import MICHAEL_DETECTOR_DIR
import os


if __name__ == '__main__':
    subj_list = [
                "raz",
                # "yon",
                "mich",
                # "noise"
    ]
    trained_on = ''
    for subj in subj_list:
        trained_on += subj + '_'

    split_ratio = 0.2
    p_components = 3
    training_window_overlap = 0.99
    inf_window_overlap = 0
    window_length = 0.3
    cooldown = 0.4

    xgb_params = {  # found with hyperparameter tuning
        'learning_rate': 0.1,
        'max_depth': 8,
        'n_estimators': 500
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    classes = [DetectionType.NEUTRAL, DetectionType.BLINK, DetectionType.GAZE_LEFT, DetectionType.GAZE_RIGHT,
                         DetectionType.GAZE_CENTER, DetectionType.GAZE_UP, DetectionType.GAZE_DOWN, DetectionType.NOISE]
    data_paths = DATA

    # init model
    model = BlinkAidXGB(classes, p_components=p_components, training_window_overlap=training_window_overlap,
                        inf_window_overlap=inf_window_overlap, window_length=window_length, cooldown=cooldown,
                        split_ratio=split_ratio, xgb_params=xgb_params)
    # train model
    model.fit(data_paths, subj_list)

    # get model performance reoprt
    acc, cm, report, report_dict = model.get_performance_report()

    # save model
    data_frac = str(int((1 - split_ratio) * 100)) + '%data_'
    model_name = trained_on + data_frac + "xgb_" + str(p_components) + 'pc'
    model_folder = str(MICHAEL_DETECTOR_DIR) + "/models/" + model_name + "_" + timestamp + "/"
    os.makedirs(model_folder, exist_ok=True)

    model_path = model_folder + model_name + "_" + timestamp + ".pkl"
    model.save(model_path)

    # save models training report
    with open(model_folder + 'classification_report.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n\n")
        f.write("Classification Report:\n")
        f.write(report)


