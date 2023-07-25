import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()
        self.query_transform = nn.Linear(d_model, d_model)
        self.key_transform = nn.Linear(d_model, d_model)
        self.value_transform = nn.Linear(d_model, d_model)

    def forward(self, input):
        # Transform the input into query, key, and value
        query = self.query_transform(input)
        key = self.key_transform(input)
        value = self.value_transform(input)

        return query, key, value

# Example usage:
d_model = 512  # The dimension of the input vector
input_vector = torch.randn(1, d_model)  # Assuming batch size 1 for simplicity

# Create an instance of the AttentionLayer
attention_layer = AttentionLayer(d_model)

# Get the query, key, and value vectors
query, key, value = attention_layer(input_vector)

# Print the shapes of the transformed vectors
print("Query shape:", query)
print("Key shape:", key)
print("Value shape:", value)

print("Query shape:", query.shape)
print("Key shape:", key.shape)
print("Value shape:", value.shape)
