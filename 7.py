import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, n_vocab, d_model, n_heads, d_ff):
        super(Transformer, self).__init__()

        self.embed = nn.Embedding(n_vocab, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff), num_layers=6
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, d_ff), num_layers=6
        )

    def forward(self, input, target):
        x = self.embed(input)
        x = self.encoder(x)
        x = self.decoder(x, target)
        return x

def main():
    transform = Transformer(1000, 128, 8, 1024)
    input = torch.randint(0, 1000, (10, 128))
    target = torch.randint(0, 1000, (10, 128))
    output = transform(input, target)
    print(output)

if __name__ == "__main__":
    main()
