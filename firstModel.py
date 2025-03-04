import pandas as pd
import numpy as np
import xgboost as xgb
import os
import plotly.graph_objects as go
from firstPlots import visualize_channels
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def preprocess_data(file_path, model_path, existing_model=False):
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
    if existing_model:
        model = xgb.XGBClassifier()
        model.load_model('23-2/processed_data.json')
        model.fit(X_train, y_train, xgb_model='23-2/processed_data.json')
    else:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)



    # Predict on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Compute confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Neutral (0)', 'Blink (1)','Gaze Left (2)','Gaze Right (3)', 'Gaze Center (4)'])

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Save the trained model
    model.save_model(model_path)
    print(f"Trained model saved to {model_path}")

    import matplotlib.pyplot as plt
    xgb.plot_importance(model)
    plt.show()

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
input_file = "data/raz_3-3/2025_03_03_1350_raz_squint.csv"  # Annotated input file
model_name = "xgb_blink_gaze1.json"
model_path = "models/" + model_name # Processed dataset for ML training
existing_model = 0

preprocess_data(input_file, model_path, existing_model)
