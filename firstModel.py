import pandas as pd
import numpy as np
import xgboost as xgb
import os
import plotly.graph_objects as go
from firstPlots import visualize_channels
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def preprocess_data(file_path, output_path):
    """Processes annotated time-series data for point-wise classification of blinks."""
    df = pd.read_csv(file_path)

    # Ensure timestamp consistency
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.drop(columns=['date', 'time'], inplace=True)

    # # Convert label to binary classification (Blink = 1, Neutral = 0)
    # df['label'] = df['label'].apply(lambda x: 1 if x == 'blink' else 0)

    # Separate features and labels
    X = df.drop(columns=['timestamp', 'label'])
    y = df['label']

    # Split data into training (first 80%) and test (last 20%)
    split_index = int(len(df) * 0.75)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Compute confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Neutral (0)', 'Blink (1)'])

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Save the trained model
    model.save_model(output_path.replace('.csv', '.json'))
    print(f"Trained model saved to {output_path.replace('.csv', '.json')}")

    # Visualize channel data using firstPlots.py
    visualize_channels_with_misclassifications(file_path, y_test, y_pred)


def visualize_channels_with_misclassifications(file_path, y_test, y_pred):
    """Visualizes channels and highlights misclassified points."""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    channels = [col for col in df.columns if col.startswith('channel_')]

    fig = go.Figure()

    # Plot all channels
    for channel in channels:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[channel],
            mode='lines',
            name=channel,
            opacity=0.5
        ))

    # Highlight misclassified points
    misclassified = y_test.to_numpy() != y_pred
    misclassified_timestamps = df['timestamp'].iloc[-len(y_test):].iloc[misclassified]
    misclassified_values = df[channels[0]].iloc[-len(y_test):].iloc[misclassified]  # Use first channel for y values

    fig.add_trace(go.Scatter(
        x=misclassified_timestamps,
        y=misclassified_values,
        mode='markers',
        marker=dict(color='red', size=8, symbol='x'),
        name='Misclassified'
    ))

    fig.update_layout(
        title='Channel Data with Misclassified Points Highlighted',
        xaxis_title='Time',
        yaxis_title='Sensor Values',
        hovermode='x unified'
    )

    fig.show()


# Example usage
input_file = "23-2/annotated/annotated_blinks.csv"  # Annotated input file
output_file = "23-2/processed_data.csv"  # Processed dataset for ML training

preprocess_data(input_file, output_file)
