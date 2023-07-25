import torch
import torch.nn as nn

class LinearModel(nn.Module):

    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input):
        x = self.linear(input)
        return x

def main():
    model = LinearModel(10, 10)
    input = torch.randn(10, 10)
    output = model(input)
    print(output)

if __name__ == "__main__":
    main()