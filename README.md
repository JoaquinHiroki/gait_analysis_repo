Gait Analysis Dementia Detection AI model

The model implements binary classification using PyTorch to analyze gait data and predict if a patient is sick or not, it takes four parameters as input which are the X,Y,Z axis and the magnitude of acceleration. The model uses a neural network with fully connected layers to process the data and then outputs a probability between 0.0 and 1.0 in which if the output is  <0.5 it is classified as healthy.

Part of the training process consists of using labeled datasets of healthy and sick patients and the loss per epoch is calculated using Binary Cross-Entropy Loss which measures how well the model's prediction matches the actual level, and a lower loss value per epoch indicates that the model is learning to predict more accurately
