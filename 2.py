import torch
import torch.nn as nn

# Input tensor with some values
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Create the ReLU activation function
relu = nn.ReLU()

# Apply the ReLU activation function to the input tensor
output = relu(x)

print(output)  # Output: tensor([0., 0., 0., 1., 2.])
