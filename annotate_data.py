import pandas as pd


def annotate_data(file_path, labels_file, output_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Ensure the dataset has a timestamp column
    if 'timestamp' not in df.columns:
        raise ValueError("Dataset must have a 'timestamp' column.")

    # Convert timestamp column to datetime and extract date and time separately
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date  # Extract date
    df['time'] = df['timestamp'].dt.strftime('%H:%M:%S.%f')  # Extract time with milliseconds
    df.drop(columns=['timestamp'], inplace=True)  # Remove original timestamp column

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

    # Save the annotated dataset
    df.to_csv(output_path, index=False)
    print(f"Annotated dataset saved to {output_path}")



annotations = {
    'raz': ['raz3-3_lc_ts.csv', 'raz3-3_lr_ts.csv', 'raz3-3_rc2_ts.csv',
            'raz3-3_ud_ts.csv', 'raz_3-3_blinks_ts.csv'],
    'yon': ['ts_blinks_yon23-2.csv', 'ts_eg1_yon23-2.csv']
}


ann_paths = {'raz': 'data/raz_3-3/', 'yon': 'data/yonatan_23-2/'}


ann_files = [ann_paths[subject] + ann for subject in annotations.keys() for ann in annotations[subject]]

data_files = []
# Example usage
data_folder_path = 'data/'
subject = 'raz_3-3/'
annotated_path = 'annotated/'
file_name = "2025_03_03_1308_raz_left_right.csv"  # Input file
labels_file = data_folder_path + subject + "raz_3-3_blinks_ts.csv"  # CSV file containing label intervals
output_path = data_folder_path + subject + annotated_path + 'annotated_' + file_name  # Output file

annotate_data(data_folder_path + subject + file_name, labels_file, output_path)
