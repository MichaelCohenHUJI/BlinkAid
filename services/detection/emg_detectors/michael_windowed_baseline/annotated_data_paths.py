from services.common.paths import PROJECT_ROOT


ann_data = {
        'raz': [
                'annotated_2025_03_03_1303_raz_blinks_no_metronome.csv',
                # 'annotated_2025_03_03_1308_raz_left_right.csv',
                'annotated_2025_03_03_1311_raz_left_center.csv',
                'annotated_2025_03_03_1319_raz_right_center_2.csv',
                'annotated_2025_03_03_1322_raz_up_down.csv',
                'annotated_2025_03_30_1301_raz_left_center.csv',
                'annotated_2025_03_30_1340_raz_right_center.csv'
                ],

        'yon': ['annotated_blinks.csv',
                # 'annotated_eye gaze left right 1.csv',
                # 'annotated_eye gaze left right 2.csv',
                'annotated_eye movements up down.csv'],

        'mich': [
                 'annotated_2025_03_03_1350_michael_blinks.csv',
                 # 'annotated_2025_03_03_1354_michael_left_right.csv',
                 'annotated_2025_03_03_1359_michael_up_down.csv',
                 # 'annotated_2025_03_30_1113_michael_left_center.csv',
                 # 'annotated_2025_03_30_1121_michael_right_center.csv',
                 # 'annotated_2025_03_30_1134_michael_left_center.csv',
                 'annotated_2025_03_30_1146_michael_left_center.csv',
                 'annotated_2025_03_30_1150_michael_right_center.csv'
        ],

        'noise': [
            'annotated_2025_03_30_1330_noise.csv'
        ]
    }

folder_paths = {'raz': '/data/raz/annotated/', 'yon': '/data/yonatan/annotated/', 'mich': '/data/michael/annotated/',
                'noise': '/data/noise/annotated/'}


DATA = {k: [str(PROJECT_ROOT) + folder_paths[k] + i for i in v] for k, v in ann_data.items()}

print(DATA)
