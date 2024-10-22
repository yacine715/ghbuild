# model_training.py

import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load selected features from JSON file
# Load and sort the data
data = pd.read_csv('data_builds_features.csv')
data['created_at'] = pd.to_datetime(data['created_at'])  # Convert created_at to datetime if it's not already
data.sort_values('created_at', inplace=True)  # Sort data chronologically

def load_selected_features(json_file='selected_features.json'):
    with open(json_file, 'r') as f:
        selected_features = json.load(f)
    return selected_features

# Model training and fine-tuning
def train_and_evaluate_model(data, selected_features):
    X = data[selected_features]  # Use only the selected features


    # Define quantiles
    quantiles = data['build_duration'].quantile([0.25, 0.75])

    # Custom function to classify duration based on quantiles
    def classify_duration_by_quantile(duration, quantiles):
        if duration <= quantiles[0.25]:
            return 'Fast'
        elif duration <= quantiles[0.75]:
            return 'Medium'
        else:
            return 'Slow'

    # Apply function to create a new column based on quantiles
    data['duration_class'] = data['build_duration'].apply(classify_duration_by_quantile, quantiles=quantiles)

    y = data['duration_class']  # Target variable

    # Handle missing values for numerical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) > 0:
        X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())  # Fill NaN with mean

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling only the numerical columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Initialize KNN model
    knn = KNeighborsClassifier()

    # Define a parameter grid to search for the best parameters for the KNN model
    param_grid = {
        'n_neighbors': range(2, 11),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Grid Search to find the best hyperparameters for classification
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters and the corresponding score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    # Save best KNN hyperparameters to a JSON file
    with open('best_knn_params.json', 'w') as f:
        json.dump(grid_search.best_params_, f)

    # Retrieve the best KNN model
    best_knn = grid_search.best_estimator_

    # Predict on the test set using the best model
    y_train_pred = best_knn.predict(X_train_scaled)
    y_test_pred = best_knn.predict(X_test_scaled)

    # Calculate accuracy, precision, recall, and F1 score for both the training and testing sets
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Print the classification results
    print(f"KNN Accuracy on test set: {test_accuracy:.2f}")
    print(f"KNN Precision on test set: {test_precision:.2f}")
    print(f"KNN Recall on test set: {test_recall:.2f}")
    print(f"KNN F1 Score on test set: {test_f1:.2f}")

    # Confusion matrix to understand the misclassifications
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:\n", conf_matrix)

    # Calculate cross-validation accuracy scores
    cv_scores = cross_val_score(best_knn, X_train_scaled, y_train, cv=5, scoring='accuracy')

    # Print the mean cross-validation score
    print("Mean cross-validation accuracy:", np.mean(cv_scores))

# Example usage:
if __name__ == "__main__":
    # Load your data here
    data = pd.read_csv('data_builds_features.csv')
    data = data[data['repo'] == 'OP-TED/eforms-notice-viewer']
    # Load the selected features from JSON
    selected_features = load_selected_features()

    # Train and evaluate the model
    train_and_evaluate_model(data, selected_features)
