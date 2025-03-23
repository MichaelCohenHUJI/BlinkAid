ann_data = {
        'raz': ['annotated_2025_03_03_1303_raz_blinks_no_metronome.csv',
                'annotated_2025_03_03_1308_raz_left_right.csv',
                'annotated_2025_03_03_1311_raz_left_center.csv',
                'annotated_2025_03_03_1319_raz_right_center_2.csv',
                'annotated_2025_03_03_1322_raz_up_down.csv'],

        'yon': ['annotated_blinks.csv',
                'annotated_eye gaze left right 1.csv',
                'annotated_eye gaze left right 2.csv',
                'annotated_eye movements up down.csv'],

        'mich': ['annotated_2025_03_03_1350_michael_blinks.csv',
                 'annotated_2025_03_03_1354_michael_left_right.csv',
                 'annotated_2025_03_03_1359_michael_up_down.csv']
    }

folder_paths = {'raz': 'data/raz_3-3/annotated/', 'yon': 'data/yonatan_23-2/annotated/', 'mich': 'data/michael_3-3/annotated/'}


DATA = {k: [folder_paths[k] + i for i in v] for k, v in ann_data.items()}

