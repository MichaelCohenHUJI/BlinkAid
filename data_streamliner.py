from pca_ica_exploration import run_pca
from windowing import create_windows


ann_data = {
        'raz': ['annotated_2025_03_03_1311_raz_left_center.csv',
                'annotated_2025_03_03_1308_raz_left_right.csv',
                'annotated_2025_03_03_1319_raz_right_center_2.csv',
                'annotated_2025_03_03_1322_raz_up_down.csv',
                'annotated_2025_03_03_1303_raz_blinks_no_metronome.csv'],

        'yon': ['annotated_blinks.csv',
                'annotated_eye gaze left right 1.csv']
    }

folder_paths = {'raz': 'data/raz_3-3/', 'yon': 'data/yonatan_23-2/'}

annotated_path = 'annotated/'

subj = 'raz'




















