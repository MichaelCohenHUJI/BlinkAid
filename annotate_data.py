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
        if labelstr == 'blink':
            label = 1
        df.loc[(df['time'] >= start_time) & (df['time'] <= stop_time), 'label'] = label

    # Save the annotated dataset
    df.to_csv(output_path, index=False)
    print(f"Annotated dataset saved to {output_path}")


# Example usage
data_folder_path = '23-2/'
annotated_path = '23-2/annotated/'
file_name = "blinks.csv"  # Input file
labels_file = "blinks_timestamps.csv"  # CSV file containing label intervals
output_path = annotated_path + 'annotated_' + file_name  # Output file

annotate_data(data_folder_path + file_name, labels_file, output_path)
