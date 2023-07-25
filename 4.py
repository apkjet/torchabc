import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1: Prepare the data
# Let's create some synthetic data for a linear regression task
np.random.seed(0)
X_train = np.random.rand(100, 1)  # 100 samples with 1 feature
y_train = 2 * X_train + 1 + np.random.randn(100, 1) * 0.1  # True relationship: y = 2x + 1 + noise

# Convert the NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Step 2: Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 input feature, 1 output (scalar)
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Step 3: Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 4: Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)

    # Compute the loss
    loss = criterion(y_pred, y_train)

    # Zero gradients, backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Test the model (Make predictions)
# For simplicity, we use the same data for testing, but in practice, we'd have separate test data.
print(y_pred)
y_pred = model(X_train)

# Print the model's learned parameters
print("Learned Parameters:")
print("Weight:", model.linear.weight.item())
print("Bias:", model.linear.bias.item())
