import os
from pca_ica_exploration import train_pca
from services.detection.emg_detectors.michael_windowed_baseline import MICHAEL_DETECTOR_DIR
from windowing import create_windows
import pandas as pd
from datetime import datetime
import joblib
from annotated_data_paths import DATA
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




def load_model(model_folder_path):
    folder_name = os.path.basename(model_folder_path)
    model = joblib.load(model_folder_path + '/' + folder_name + '.pkl')
    model_metadata = joblib.load(model_folder_path + '/' + folder_name + '_metadata.pkl')
    scaler = joblib.load(model_metadata['scaler_path'])
    pca = joblib.load(model_metadata['pca_model_path'])
    p_components = model_metadata['p_components']
    return model, scaler, pca, p_components

def apply_pca(df, scaler, pca, p_components):
    """
    Apply the saved scaler and PCA to the dataframe.
    Assumes the dataframe has columns 'timestamp', sensor columns, and 'label'.
    """
    pca_cols = [f'PC{i+1}' for i in range(p_components)]
    # Assume sensor columns are all except 'timestamp' and 'label'
    sensor_columns = [col for col in df.columns if col not in ['timestamp', 'label']]
    # Scale the sensor data
    scaled = pd.DataFrame(scaler.transform(df[sensor_columns]), columns=sensor_columns)
    # Apply PCA on the scaled data
    pcaed = pd.DataFrame(pca.transform(scaled), columns=pca_cols)
    # Reattach timestamp and label columns
    pcaed['timestamp'] = df['timestamp'].values
    pcaed['label'] = df['label'].values
    # Ensure column order (timestamp, PCA components, label)
    cols = ['timestamp'] + pca_cols + ['label']
    return pcaed[cols]



def main(model_folder_path, data_paths, window_length=0.3, overlap=0.99):
    # Load the existing model components
    print("Loading model components...")
    model, scaler, pca, p_components = load_model(model_folder_path)
    p_components = p_components

    # Collect annotated input data
    test_dfs = []
    for file_path in data_paths:
        print(f"Reading {file_path}...")
        df = pd.read_csv(file_path)
        test_dfs.append(df)

    # Apply PCA transformation to each annotated dataset
    print("Applying PCA transformation...")
    test_dfs_pca = [apply_pca(df, scaler, pca, p_components) for df in test_dfs]

    # Create windows from the PCA-transformed data
    print("Creating windows from annotated data...")
    test_windows = []
    for df in tqdm(test_dfs_pca, desc="Windowing"):
        windows = create_windows(df, window_length, overlap)
        test_windows.append(windows)
    test_windows_df = pd.concat(test_windows, ignore_index=True)

    # Prepare the features for prediction by dropping non-feature columns
    # Assumes that 'timestamp' and 'label' are not part of the features.
    X_test = test_windows_df.drop(columns=['timestamp', 'label'])
    y_test = test_windows_df['label']

    # Get predictions from the existing model
    print("Making predictions on new data...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Compute confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Neutral (0)', 'Blink (1)', 'Gaze Left (2)',
                                                                 'Gaze Right (3)', 'Gaze Center (4)', 'Gaze Up (5)',
                                                                 'Gaze Down (6)'])
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # with open(model_folder_path + 'classification_report.txt', 'w') as f:
    #     f.write("Confusion Matrix:\n")
    #     f.write(str(cm) + "\n\n\n")
    #     f.write("Classification Report:\n")
    #     f.write(report)

    # # Add predictions to the dataframe
    # test_windows_df['prediction'] = y_pred
    #
    # # Save predictions to CSV
    # output_path = os.path.join(model_folder_path, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    # test_windows_df.to_csv(output_path, index=False)
    # print(f"Predictions saved to {output_path}")



if __name__ == '__main__':
    model_folder_path = str(MICHAEL_DETECTOR_DIR) + "/models/"
    model_name = 'raz_20%data_xg_windowed_stdized_3pc_2025-03-23_16-15-36'
    data_paths = DATA['raz']
    main(model_folder_path + model_name, data_paths)