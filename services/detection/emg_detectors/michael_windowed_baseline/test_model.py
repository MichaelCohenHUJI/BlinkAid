from services.detection.emg_detectors.michael_windowed_baseline.BlinkAidXGB import BlinkAidXGB
from services.detection.emg_detectors.michael_windowed_baseline import MICHAEL_DETECTOR_DIR
from annotated_data_paths import DATA


if __name__ == '__main__':
    data_paths = DATA
    subj_list = [
                "raz",
                "yon",
                "mich",
                "noise"
    ]

    models_path = str(MICHAEL_DETECTOR_DIR) + "/models/"
    model_name = "raz_yon_mich_noise_80%data_xgb_3pc_2025-04-09_15-07-56"
    model = BlinkAidXGB.load(models_path + model_name + '/' + model_name + '.pkl')
    report = model.test_model(data_paths, subj_list)
    model_folder = str(MICHAEL_DETECTOR_DIR) + "/models/" + model_name + "/"

    # save models training report
    with open(model_folder + 'Test_report.txt', 'w') as f:
        f.write("Test results:\n")
        f.write(str(report) + "\n\n\n")



