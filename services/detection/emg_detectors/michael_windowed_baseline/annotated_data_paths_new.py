from services.common.paths import PROJECT_ROOT


ann_data = {
        'raz': [
                'annotated_data_20250901_171720_converted.csv',
                'annotated_data_20250901_180608_converted.csv',
                'annotated_data_20250901_183658_converted.csv',
                'annotated_data_20250901_184236_converted.csv',
                ],

        'yech': [
                'annotated_data_20250901_160603_converted.csv',
                'annotated_data_20250901_160831_converted.csv',
                'annotated_data_20250901_162131_converted.csv',
                'annotated_data_20250901_163125_converted.csv',
                ],

        'shir': [
                 'annotated_data_20250901_195658_converted.csv',
                 'annotated_data_20250901_195908_converted.csv',
                 'annotated_data_20250901_200010_converted.csv',
                ],

        'noise': [
                 'annotated_data_20250901_185926_converted.csv',
                ]
    }

folder_paths = {'raz': '/data/new/raz/annotated/', 'yech': '/data/new/yechiam/annotated/', 'shir': '/data/new/shir/annotated/',
                'noise': '/data/new/noise/annotated/'}


DATA = {k: [str(PROJECT_ROOT) + folder_paths[k] + i for i in v] for k, v in ann_data.items()}

print(DATA)
