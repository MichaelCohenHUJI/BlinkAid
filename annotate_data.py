import pandas as pd


def annotate_data(file_path, label_intervals, output_path):
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
    df['label'] = 'neutral'

    # Iterate through the label intervals and update the Label column
    for start_time, stop_time, label in label_intervals:
        start_time = pd.to_datetime(start_time).strftime('%H:%M:%S.%f')
        stop_time = pd.to_datetime(stop_time).strftime('%H:%M:%S.%f')
        df.loc[(df['time'] >= start_time) & (df['time'] <= stop_time), 'label'] = label

    # Save the annotated dataset
    df.to_csv(output_path, index=False)
    print(f"Annotated dataset saved to {output_path}")


# Example usage
data_folder_path = '23-2/'
annotated_path = '23-2/annotated/'
file_name = "blinks.csv"  # Input file
output_path = annotated_path + 'annotated_' + file_name  # Output file

# Example list of intervals to annotate (preserving milliseconds)
label_intervals = [
    ("14:09:55.689", "14:09:57.500", "blink"),
    ("14:10:05.123", "14:10:06.789", "gazing_left"),
    ("14:10:08.456", "14:10:09.987", "gazing_right")
]

annotate_data(data_folder_path + file_name, label_intervals, output_path)