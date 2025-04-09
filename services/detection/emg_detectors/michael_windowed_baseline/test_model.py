from services.detection.emg_detectors.michael_windowed_baseline.BlinkAidXGB import BlinkAidXGB
from services.detection.emg_detectors.michael_windowed_baseline import MICHAEL_DETECTOR_DIR
from annotated_data_paths import DATA


if __name__ == '__main__':
    data_paths = DATA
    subj_list = [
                # "raz",
                "yon",
                # "mich"
    ]

    models_path = str(MICHAEL_DETECTOR_DIR) + "/models/"
    model_name = "raz_mich_80%data_xgb_3pc_2025-04-08_16-19-29"
    model = BlinkAidXGB.load(models_path + model_name + '/' + model_name + '.pkl')
    model.test_model(data_paths, subj_list)
