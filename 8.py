from transformers import BertTokenizer, BertModel
import torch

# Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Input text
text = "Hello, how are you doing today?"

# Tokenize the input text
tokens = tokenizer(text, return_tensors='pt')

# Forward pass through the model to get the embeddings
outputs = model(**tokens)

# Get the pooled output (CLS token) and the last layer hidden states
pooled_output = outputs['pooler_output']
last_hidden_states = outputs['last_hidden_state']

# Print the pooled output (CLS token)
print("Pooled Output:")
print(pooled_output)

# Print the last layer hidden states
print("\nLast Layer Hidden States:")
print(last_hidden_states)
