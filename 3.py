import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1: Prepare the data
# Let's create some synthetic data for a binary classification task
np.random.seed(0)
X_train = np.random.rand(100, 2)  # 100 samples with 2 features
y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)  # Binary labels based on the sum of features

# Convert the NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Step 2: Define the neural network model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(2, 2)  # 2 input features, 2 output classes (binary)
    def forward(self, x):
        return self.fc(x)

model = SimpleClassifier()



# Step 3: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification tasks
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
y_pred_probs = torch.softmax(model(X_train), dim=1)
y_pred_labels = torch.argmax(y_pred_probs, dim=1)

print("Predicted Probabilities:")
print(y_pred_probs)
print("Predicted Labels:")
print(y_pred_labels)
