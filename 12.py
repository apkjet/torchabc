import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # The forward function defines the computation performed by the model
        return self.linear(x)

# Example usage:
input_size = 10
output_size = 1
model = LinearModel(input_size, output_size)

# Random input tensor of size (batch_size, input_size)
batch_size = 5
input_tensor = torch.randn(batch_size, input_size)

# Make a forward pass through the model
output_tensor = model(input_tensor)

print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
