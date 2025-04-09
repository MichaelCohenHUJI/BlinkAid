from datetime import datetime
from services.detection.emg_detectors.michael_windowed_baseline.BlinkAidXGB import BlinkAidXGB
from services.detection.emg_detectors.michael_windowed_baseline import MICHAEL_DETECTOR_DIR
from annotated_data_paths import DATA
import os
import re



if __name__ == '__main__':
    data_paths = DATA
    subj_list = [
                # "raz",
                "yon",
                # "mich"
                # "noise"
    ]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    models_path = str(MICHAEL_DETECTOR_DIR) + "/models/"
    model_name = "raz_mich_80%data_xgb_3pc_2025-04-08_16-19-29"
    model = BlinkAidXGB.load(models_path + model_name + '/' + model_name + '.pkl')
    model.continue_fit(data_paths, subj_list)
    # get model performance reoprt
    acc, cm, report, report_dict = model.get_performance_report()

    # save model
    old_name = model_name
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    match = re.search(date_pattern, model_name)
    if match:
        model_name = model_name[:match.start()]
    model_folder = str(MICHAEL_DETECTOR_DIR) + "/models/" + model_name + "_cont_" + timestamp + "/"
    os.makedirs(model_folder, exist_ok=True)

    model_path = model_folder + model_name + "_cont_" + timestamp + ".pkl"
    model.save(model_path)

    # save models training report
    with open(model_folder + 'classification_report.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("Trained from previous model:\n")
        f.write(old_name)
