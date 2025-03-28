import pandas as pd
import os

def annotate_data(file_path, labels_file, output_path):
    '''
    gets the raw data and labels timestamps and creates a labeled df. saving as a file and returns it
    :param file_path: raw data path
    :param labels_file: labels timestamps path
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
    # df.drop(columns=['timestamp'], inplace=True)  # Remove original timestamp column

    # Initialize the 'Label' column with 'neutral'
    df['label'] = 0

    # Load label intervals from the CSV file
    label_intervals = pd.read_csv(labels_file)

    # Iterate through the label intervals and update the Label column
    for _, row in label_intervals.iterrows():
        start_time = row['start']
        stop_time = row['stop']
        labelstr = row['label']
        label = 0
        if labelstr == 'garbage':
            df = df.drop(df[(df['time'] >= start_time) & (df['time'] <= stop_time)].index)
        elif labelstr == 'blink':
            label = 1
        elif labelstr == 'gazeleft':
            label = 2
        elif labelstr == 'gazeright':
            label = 3
        elif labelstr == 'gazecenter':
            label = 4
        elif labelstr == 'gazeup':
            label = 5
        elif labelstr == 'gazedown':
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
        'raz': [('2025_03_03_1311_raz_left_center.csv', 'raz3-3_lc_ts.csv'),
                ('2025_03_03_1308_raz_left_right.csv', 'raz3-3_lr_ts.csv'),
                ('2025_03_03_1319_raz_right_center_2.csv', 'raz3-3_rc2_ts.csv'),
                ('2025_03_03_1322_raz_up_down.csv', 'raz3-3_ud_ts.csv'),
                ('2025_03_03_1303_raz_blinks_no_metronome.csv', 'raz_3-3_blinks_ts.csv')],

        'yon': [('blinks.csv', 'ts_blinks_yon23-2.csv'),
                ('eye gaze left right 1.csv', 'ts_eg1_yon23-2.csv'),
                ('eye gaze left right 2.csv', 'ts_eg2_yon23-2.csv'),
                ('eye movements up down.csv', 'ts_ud_yon23-2.csv')],

        'mich': [('2025_03_03_1350_michael_blinks.csv', 'michael_3-3_blinks_ts.csv'),
                 ('2025_03_03_1354_michael_left_right.csv', 'michael_3-3_lr_ts.csv'),
                 ('2025_03_03_1359_michael_up_down.csv', 'michael_3-3_ud_ts.csv')]
    }

    folder_paths = {'raz': 'data/raz_3-3/', 'yon': 'data/yonatan_23-2/', 'mich': 'data/michael_3-3/'}

    annotated_path = 'annotated/'
    for subj in data_ann_pairs.keys():
        for pair in data_ann_pairs[subj]:
            data_file_name, label_file = pair
            folder_path = folder_paths[subj]
            output_name = 'annotated_' + data_file_name
            out_path = folder_path + annotated_path + output_name
            if not os.path.exists(out_path):
                annotate_data(folder_path + data_file_name, folder_path + label_file, out_path)
