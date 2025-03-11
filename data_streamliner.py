from pca_ica_exploration import run_pca
from windowing import create_windows
import pandas as pd
from firstModel import preprocess_data


if __name__ == '__main__':
        ann_data = {
                'raz': ['annotated_2025_03_03_1303_raz_blinks_no_metronome.csv',
                        'annotated_2025_03_03_1308_raz_left_right.csv',
                        'annotated_2025_03_03_1311_raz_left_center.csv',
                        'annotated_2025_03_03_1319_raz_right_center_2.csv',
                        'annotated_2025_03_03_1322_raz_up_down.csv'],

                'yon': ['annotated_blinks.csv',
                        'annotated_eye gaze left right 1.csv']
            }

        folder_paths = {'raz': 'data/raz_3-3/annotated/', 'yon': 'data/yonatan_23-2/annotated/'}

        subj = 'raz'

        # collect data from all files
        ann_data_paths = [folder_paths[subj] + f for f in ann_data[subj]]
        df = pd.concat((pd.read_csv(f) for f in ann_data_paths), ignore_index=True)
        p_components = 16
        df, pca_results, scaler = run_pca(df, p_components)
        windows = create_windows(df, 0.4)
        model_name = "naive_xg_windowed"
        existing_model = 0
        n = 7

        preprocess_data(df, n, model_name=model_name)



















