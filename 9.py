import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define a simple MLP classifier
class MNISTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Dummy MNIST data
mnist_data = torch.randn(100, 28*28)  # 100 samples with 28*28 features (flattened image)

# Tokenize the input data
tokens = tokenizer.batch_encode_plus(mnist_data.tolist(), padding=True, return_tensors='pt')

# Forward pass through the transformer model
outputs = model(**tokens)

# Get the pooled output (CLS token) from the transformer model
pooled_output = outputs['pooler_output']

# Define the classifier and pass the pooled output through it
classifier = MNISTClassifier(input_size=768, hidden_size=256, output_size=10)
logits = classifier(pooled_output)

# Print the logits (output scores) for each class
print("Logits:")
print(logits)
