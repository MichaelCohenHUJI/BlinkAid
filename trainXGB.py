import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_xgb(traindf, testdf, n_classes, classes_strings):
    """

    :param traindf:
    :param testdf:
    :param n_classes:
    :return:
    """

    traindf = traindf.sample(frac=1).reset_index(drop=True)
    testdf = testdf.sample(frac=1).reset_index(drop=True)
    # Separate features and labels
    X_train = traindf.drop(columns=['timestamp', 'label'])
    y_train = traindf['label']
    X_test = testdf.drop(columns=['timestamp', 'label'])
    y_test = testdf['label']

    # Train XGBoost model
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
    report = classification_report(y_test, y_pred, target_names=classes_strings)
    report_dict = classification_report(y_test, y_pred, target_names=classes_strings, output_dict=True)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return model, cm, report, report_dict
