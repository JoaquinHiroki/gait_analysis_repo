import pandas as pd
import numpy as np

# Function to generate synthetic data
def generate_gait_data(num_samples, x_range, y_range, z_range, acc_range):
    data = {
        "X": np.random.uniform(x_range[0], x_range[1], num_samples),
        "Y": np.random.uniform(y_range[0], y_range[1], num_samples),
        "Z": np.random.uniform(z_range[0], z_range[1], num_samples),
    }
    data["Accelerometer"] = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2) + np.random.uniform(
        acc_range[0], acc_range[1], num_samples
    )
    return pd.DataFrame(data)

# Generate data for healthy patients
healthy_data = generate_gait_data(
    num_samples=1000, 
    x_range=(-1, 1), 
    y_range=(-1, 1), 
    z_range=(-1, 1), 
    acc_range=(0, 0.5)
)

# Generate data for sick patients
sick_data = generate_gait_data(
    num_samples=1000, 
    x_range=(-3, 3), 
    y_range=(-3, 3), 
    z_range=(-3, 3), 
    acc_range=(0.5, 1.5)
)

# Save to CSV
healthy_data.to_csv("healthy_patients.csv", index=False)
sick_data.to_csv("sick_patients.csv", index=False)

print("Synthetic data generated and saved to 'healthy_patients.csv' and 'sick_patients.csv'.")
