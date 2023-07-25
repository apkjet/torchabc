import torch
import torch.nn as nn

class SimpleFullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # First fully connected layer
        self.relu = nn.ReLU()                           # Activation function (ReLU)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer

    def forward(self, x):
        x = self.fc1(x)      # Linear transformation in the first layer
        x = self.relu(x)     # Activation function (ReLU)
        x = self.fc2(x)      # Linear transformation in the second layer
        return x

# Example usage:
input_size = 10    # Number of input features
hidden_size = 20   # Number of neurons in the hidden layer
output_size = 2    # Number of output classes (binary classification)

# Create an instance of the simple fully connected neural network
model = SimpleFullyConnectedNN(input_size, hidden_size, output_size)

# Print the model architecture
print(model)

