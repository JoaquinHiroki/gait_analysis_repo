#Joaquin Hiroki Campos Kishi A01639134
#November 15, 2024
#File to train an AI model using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

#Create a custom dataset
class GaitDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

#Neural Network 
class GaitAnalysisModel(nn.Module):
    def __init__(self):
        super(GaitAnalysisModel, self).__init__()
        self.fc = nn.Sequential(

            nn.Linear(4, 32), #Feautures 
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  #Binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

#Load Data from CSV Files
healthy_data = pd.read_csv("healthy_patients.csv")  # Healthy patients
sick_data = pd.read_csv("sick_patients.csv")  # Sick patients

#Combine and Label Data
healthy_data["label"] = 0  # Label healthy as 0
sick_data["label"] = 1  # Label sick as 1
combined_data = pd.concat([healthy_data, sick_data], ignore_index=True)

#Shuffle Data
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

#Split into Features (X) and Labels (y)
X = combined_data[["X", "Y", "Z", "Accelerometer"]].values
y = combined_data["label"].values

# Train-Test Split
train_data, test_data = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
train_labels, test_labels = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

# Create DataLoader
train_dataset = GaitDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize Model, Loss Function, Optimizer
model = GaitAnalysisModel()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data).squeeze()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "gait_analysis_model.pth")
print("Model saved successfully.")

