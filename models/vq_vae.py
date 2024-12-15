import torch
from torch.nn import Module, Sequential, ReLU, Conv2d, ConvTranspose2d, Embedding
from torchsummary import summary


class VQVAE(Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_embeddings, embedding_dim):
        super().__init__()

        self.encoder = Sequential(
            Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Conv2d(hidden_dim * 2, embedding_dim, kernel_size=4, stride=2, padding=1)
        )

        self.decoder = Sequential(
            ConvTranspose2d(embedding_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)
        )

        self.embedding = Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e):
        flat_z_e = z_e.view(-1, self.embedding_dim)
        distances = torch.cdist(flat_z_e, self.embedding.weight)
        encoding_indices = torch.argmin(distances, dim=-1)

        z_q = self.embedding(encoding_indices)
        z_q = z_q.view(z_e.shape)

        return z_q, encoding_indices

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, encoding_indices = self.quantize(z_e)
        x_recon = self.decoder(z_q)

        return x_recon, z_e, z_q, encoding_indices

if __name__ == '__main__':
    model = VQVAE(3, 1, hidden_dim=64, num_embeddings=512, embedding_dim=64)

    summary(model, (3, 256, 256))
