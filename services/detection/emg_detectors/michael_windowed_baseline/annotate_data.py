import pandas as pd
import os
from services.common.paths import PROJECT_ROOT


def annotate_data(file_path, labels_file_path, output_path):
    '''
    gets the raw data and labels timestamps and creates a labeled df. saving as a csv file and returns it as df
    :param file_path: raw data path
    :param labels_file_path: labels timestamps path
    :param output_path: where to save the labeled data (including new file name)
    :return: the labeled df of the format: timestamp, channel_1-channel_16, label
    '''
    # Load the dataset
    df = pd.read_csv(file_path)

    # Ensure the dataset has a timestamp column
    if 'timestamp' not in df.columns:
        raise ValueError("Dataset must have a 'timestamp' column.")

    # Convert timestamp column to datetime and extract date and time separately
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'].dt.date  # Extract date
    df['time'] = df['timestamp'].dt.strftime('%H:%M:%S.%f')  # Extract time with milliseconds

    # Initialize the 'Label' column with 'neutral'
    df['label'] = 0   # neutral label is 0

    # Load label intervals from the CSV file
    label_intervals = pd.read_csv(labels_file_path)

    # Iterate through the label intervals and update the Label column
    for _, row in label_intervals.iterrows():
        start_time = row['start']
        stop_time = row['end']
        labelstr = row['label']
        label = 0
        if labelstr == 'garbage':
            df = df.drop(df[(df['time'] >= start_time) & (df['time'] <= stop_time)].index)
        elif labelstr == 'blink':
            label = 1
        elif labelstr == 'gaze_left':
            label = 2
        elif labelstr == 'gaze_right':
            label = 3
        elif labelstr == 'gaze_up':
            label = 4
        elif labelstr == 'gaze_down':
            label = 5
        elif labelstr == 'noise':
            label = 6
        else:
            print(f"Unknown label: {labelstr}, in timestamp: {start_time} - {stop_time}")
            return
        df.loc[(df['time'] >= start_time) & (df['time'] <= stop_time), 'label'] = label

    df.drop(columns=['time', 'date'], inplace=True)

    # Save the annotated dataset
    df.to_csv(output_path, index=False)
    print(f"Annotated dataset saved to {output_path}")

    return df


if __name__ == '__main__':
    data_ann_pairs = {
        'raz': [('data_20250901_171720_converted.csv', 'data_20250901_171720_converted_ts.csv'),
                # ('data_20250901_173420_converted.csv', 'data_20250901_173420_converted_ts.csv'),
                ('data_20250901_180608_converted.csv', 'data_20250901_180608_converted_ts.csv'),
                ('data_20250901_183658_converted.csv', 'data_20250901_183658_converted_ts.csv'),
                ('data_20250901_184236_converted.csv', 'data_20250901_184236_converted_ts.csv'),
                ('data_20250901_185926_converted.csv', 'data_20250901_185926_converted_ts.csv')],

        'shir': [('data_20250901_195658_converted.csv', 'data_20250901_195658_converted_ts.csv'),
                ('data_20250901_195908_converted.csv', 'data_20250901_195908_converted_ts.csv'),
                ('data_20250901_200010_converted.csv', 'data_20250901_200010_converted_ts.csv')],

        'yech': [('data_20250901_160603_converted.csv', 'data_20250901_160603_converted_ts.csv'),
                 ('data_20250901_160831_converted.csv', 'data_20250901_160831_converted_ts.csv'),
                 ('data_20250901_162131_converted.csv', 'data_20250901_162131_converted_ts.csv'),
                 ('data_20250901_163125_converted.csv', 'data_20250901_163125_converted_ts.csv')],

        # 'noise': [('2025_03_30_1330_noise.csv', 'noise_30-3_ts.csv')]
    }

    folder_paths = {'raz': 'data/new/raz/', 'shir': 'data/new/shir/', 'yech': 'data/new/yechiam/',
                    #'noise': 'data/noise/'
                    }

    annotated_path = 'annotated/'
    for subj in data_ann_pairs.keys():
        for pair in data_ann_pairs[subj]:
            data_file_name, label_file = pair
            folder_path = str(PROJECT_ROOT) + '/' + folder_paths[subj]
            if not os.path.exists(folder_path + annotated_path):
                os.makedirs(folder_path + annotated_path)
            output_name = 'annotated_' + data_file_name
            out_path = folder_path + annotated_path + output_name
            if not os.path.exists(out_path) or True:
                annotate_data(folder_path + data_file_name, folder_path + label_file, out_path)

