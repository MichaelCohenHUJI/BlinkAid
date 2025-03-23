import pandas as pd
import numpy as np
import xgboost as xgb
import os
import plotly.graph_objects as go
from firstPlots import visualize_channels
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




def train_xgb(traindf, testdf, n_classes, model_path=None, existing_model=False):
    """Processes annotated time-series data for point-wise classification of blinks."""

    # Ensure timestamp consistency
    # df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    # df.drop(columns=['date', 'time'], inplace=True)

    traindf = traindf.sample(frac=1).reset_index(drop=True)
    testdf = testdf.sample(frac=1).reset_index(drop=True)
    # Separate features and labels
    X_train = traindf.drop(columns=['timestamp', 'label'])
    y_train = traindf['label']
    X_test = testdf.drop(columns=['timestamp', 'label'])
    y_test = testdf['label']


    # Train XGBoost model
    if existing_model:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        model.fit(X_train, y_train, xgb_model=model_path)
    else:
        model = xgb.XGBClassifier(
            eval_metric='mlogloss',  # Multi-class log loss
            objective='multi:softprob',  # Softmax output
            num_class=n_classes,  # Replace N with the number of classes
        )
        model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Compute confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Neutral (0)', 'Blink (1)','Gaze Left (2)',
                                                                 'Gaze Right (3)', 'Gaze Center (4)', 'Gaze Up (5)',
                                                                 'Gaze Down (6)'])
    report_dict = classification_report(y_test, y_pred, target_names=['Neutral (0)', 'Blink (1)', 'Gaze Left (2)',
                                                                 'Gaze Right (3)', 'Gaze Center (4)', 'Gaze Up (5)',
                                                                 'Gaze Down (6)'], output_dict=True)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    import matplotlib
    matplotlib.use("TkAgg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    xgb.plot_importance(model)
    plt.show()

    return model, cm, report, report_dict

    # Visualize channel data using firstPlots.py
    # visualize_channels_with_misclassifications(datadf, y_test, y_pred)


def visualize_channels_with_misclassifications(df, y_test, y_pred):
    """Visualizes channels and highlights misclassified points."""
    # df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    channels = [col for col in df.columns[1:-1]]

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


if __name__ == '__main__':
    data_files = {
        'raz': ['2025_03_03_1303_raz_blinks_no_metronome.csv',
                '2025_03_03_1308_raz_left_right.csv',
                '2025_03_03_1311_raz_left_center.csv',
                '2025_03_03_1319_raz_right_center_2.csv',
                '2025_03_03_1322_raz_up_down.csv'],

        'yon': ['blinks.csv',
                'eye gaze left right 1.csv']
    }

    data_paths = {'raz': 'data/raz_3-3/annotated/annotated_', 'yon': 'data/yonatan_23-2/annotated/annotated_',
                  'michael': 'data/michael_3-3/'}

    subj = 'raz'

    data_files_paths = [data_paths[subj] + f for f in data_files[subj]]
    df = pd.concat((pd.read_csv(f) for f in data_files_paths), ignore_index=True)

    model_name = "naive_xg"
    existing_model = 0
    n = 7

    train_xgb(df, n, model_name=model_name)
