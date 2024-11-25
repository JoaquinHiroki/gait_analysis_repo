# Joaquin Hiroki Campos Kishi A01639134
# November 15, 2024
# File to make predictions using a trained AI model

import torch
import pandas as pd
import numpy as np
from training import GaitAnalysisModel  # Ensure the `training.py` file is in the same directory

# Load the trained model
model = GaitAnalysisModel()
model.load_state_dict(torch.load("gait_analysis_model.pth"))
model.eval()

# Function to preprocess input data
def preprocess_input(data):
    # Ensure the input is a numpy array
    data = np.array(data, dtype=np.float32)
    # Convert to a PyTorch tensor
    return torch.tensor(data, dtype=torch.float32)

# Function to make predictions
def predict(data):
    data = preprocess_input(data)
    with torch.no_grad():
        outputs = model(data).squeeze()
        predictions = (outputs >= 0.5).int()  # Convert probabilities to binary predictions
        return predictions.numpy(), outputs.numpy()  # Return both binary predictions and probabilities

# Load new data for prediction
# Assuming a CSV file with the same feature columns as training data
new_data = pd.read_csv("new_gait_data.csv")  # Replace with your actual file name
X_new = new_data[["X", "Y", "Z", "Accelerometer"]].values

# Make predictions
binary_predictions, probabilities = predict(X_new)

# Add predictions to the dataframe
new_data["Prediction"] = binary_predictions
new_data["Probability"] = probabilities

# Save predictions to a new CSV file
new_data.to_csv("predictions.csv", index=False)
print("Predictions saved to 'predictions.csv'.")
