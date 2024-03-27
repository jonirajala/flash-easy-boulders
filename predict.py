import torch
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import math

from utils import load_data_from_db, preprocess_data, get_features, SimpleMLPRegression, initialize_database_if_needed

MODEL_PATH = "model.pth"
N_TOP_DIFFERENCES = 20

def load_and_prepare_model(model_path):
    """Load the pretrained model from the given path."""
    if not os.path.exists(model_path):
        print(f"Error: Model file at {model_path} not found.")
        sys.exit(1)  # Exit the program with an error status.

    model = SimpleMLPRegression(1787)  # Assuming 'SimpleMLPRegression' is defined in 'utils'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")
    return model

def preprocess_and_extract_features():
    """Load, preprocess data, and extract features."""
    print("Loading data")
    holes_df, routes_df, routes_grade_df, grade_comparision_df = load_data_from_db()
    grade_comparision = grade_comparision_df.set_index('difficulty')['boulder_name'].to_dict()
    print("Data Loaded")

    # Only include boulders that are publicly listed in app
    print(f"Excluding boulders that are not publicly listed, size before: {len(routes_df)}")
    routes_df = routes_df[routes_df['is_listed'] == 1]
    print(f"Size after: {len(routes_df)}")

    print("Preprocessing data")
    routes_l1 = preprocess_data(routes_df, routes_grade_df, ascents=2)
    print(f"Data preprocessed, {len(routes_l1)} boulder problems")

    print("Extracting features")
    routes = get_features(routes_l1, holes_df)
    print(f"Features extracted. Got {routes.shape} features")

    
    return routes, routes_l1, routes_df, grade_comparision

def prepare_features_for_model(routes, routes_l1):
    """Prepare and scale features for model input."""
    print("Preparing features")
    N, H, W = routes.shape
    features = np.hstack([routes.reshape(N, -1), routes_l1['angle'].values.reshape(-1, 1)])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return torch.tensor(features, dtype=torch.float), routes_l1['uuid']

def make_predictions(model, features, actual_values):
    """Make predictions using the model and compare with actual values."""
    print("Making Predictions")
    with torch.no_grad():  # Disable gradient computation
        predictions = model(features).squeeze()
    differences = (actual_values - predictions)
    return predictions, differences

def display_top_differences(differences, uuids, actual_values, predictions, routes_df, grade_comparision, N=N_TOP_DIFFERENCES):
    """Display top N differences between actual and predicted values."""
    differences_np = differences.cpu().numpy()
    top_indices = np.argsort(differences_np)[-N:]
    print(f"Top {N} datapoints with the biggest prediction differences:")
    for rank, index in enumerate(reversed(top_indices), start=1):
        uuid = uuids[index]
        print(f"\nRank {rank}, Datapoint {index}, UUID: {uuid}")
        print(f"Actual: {actual_values[index].item():.2f} ({grade_comparision[math.floor(actual_values[index].item())]}), Predicted: {predictions[index].item():.2f} ({grade_comparision[math.floor(predictions[index].item())]}), Difference: {differences[index].item():.2f}")
        subset = routes_df[routes_df['uuid'] == uuid][['setter_username', 'name']]
        print(subset.to_string(index=False))

def main():
    initialize_database_if_needed()

    model = load_and_prepare_model(MODEL_PATH)
    routes, routes_l1, routes_df, grade_comparision = preprocess_and_extract_features()
    features, uuids = prepare_features_for_model(routes, routes_l1)
    actual_values = torch.tensor(routes_l1['difficulty_average'].values, dtype=torch.float)
    predictions, differences = make_predictions(model, features, actual_values)
    display_top_differences(differences, uuids, actual_values, predictions, routes_df, grade_comparision)

if __name__ == "__main__":
    main()
