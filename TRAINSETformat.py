import pandas as pd
import numpy as np
from datetime import datetime


def transform_csv(input_file):
    """
    Transform a CSV with timestamp and 16 channels into a 4-column format.
    Converts timestamps to ISO 8601 format.

    Args:
        input_file (str): Path to input CSV file
    """
    print(f"Reading {input_file}...")
    # Read the CSV file
    path = './23-2/'
    df = pd.read_csv(path + input_file)

    # Create a list to hold the transformed data
    transformed_data = []

    # Get total number of rows for progress tracking
    total_rows = len(df)
    print(f"Processing {total_rows} rows...")

    # Process each row
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{total_rows}...")

        # Get the timestamp and convert to ISO 8601 format
        try:
            # Parse timestamp based on expected format (adjust if needed)
            # Assuming format is like: '2025-02-23 14:14:51.478000'
            timestamp_str = row['timestamp']
            dt = pd.to_datetime(timestamp_str)
            iso_timestamp = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        except Exception as e:
            print(f"Error converting timestamp at row {idx}: {e}")
            # If conversion fails, use original timestamp
            iso_timestamp = row['timestamp']

        # For each channel, create a new row
        for i in range(1, 17):
            channel_name = f"channel_{i}"
            value = row[channel_name]

            # Create a new row with series, timestamp, value, and empty label
            transformed_data.append({
                'series': channel_name,
                'timestamp': iso_timestamp,
                'value': value,
                'label': ''  # Empty label column
            })

    # Create a DataFrame from the transformed data
    transformed_df = pd.DataFrame(transformed_data)

    # Write to CSV
    output_file = 'TRAINSET ' + input_file
    print(f"Writing {len(transformed_df)} rows to {output_file}...")
    transformed_df.to_csv(output_file, index=False)
    print("Transformation complete!")

    return transformed_df


# Example usage:
transform_csv('eye gaze left right 1.csv')