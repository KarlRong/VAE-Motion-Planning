import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F


class convVAE(nn.Module):
    def __init__(self, sample_size, grid_size, cnnout_size, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super(convVAE, self).__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.condnn = CondNN(sample_size, grid_size, cnnout_size)
        self.encoder = Encoder(sample_size * 3 + cnnout_size, encoder_layer_sizes, latent_size)
        self.decoder = Decoder(latent_size + sample_size * 2 + cnnout_size, decoder_layer_sizes, sample_size)

    def cnn(self, startend, occ):
        return self.cnn(startend, occ)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, startend, occ):
        c = self.condnn(startend, occ)
        mu, logvar = self.encode(torch.cat((x, c), dim=-1))
        z = self.reparameterize(mu, logvar)
        return self.decode(torch.cat((z, c), dim=-1)), mu, logvar

    def inference(self, startend, occ, num_viz):
        c = self.condnn(startend, occ)
        z = torch.randn(num_viz, self.latent_size, device=c.device)
        return self.decode(torch.cat((z, c), dim=-1))


class Encoder(nn.Module):
    def __init__(self, input_size, layer_sizes, latent_size):
        super(Encoder, self).__init__()

        layer_sizes = [input_size] + layer_sizes
        modules = []
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            modules.append(nn.Linear(in_size, out_size))
            modules.append(nn.ReLU())
        #             modules.append(nn.Dropout(p=0.5))

        self.sequential = nn.Sequential(*modules)
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        x = self.sequential(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, input_size, layer_sizes, sample_size):
        super(Decoder, self).__init__()

        layer_sizes = [input_size] + layer_sizes
        modules = []
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            modules.append(nn.Linear(in_size, out_size))
            modules.append(nn.ReLU())
        #             modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(layer_sizes[-1], sample_size))

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)


class CondNN(nn.Module):
    def __init__(self, sampleSize, gridSize, outSize):
        super(CondNN, self).__init__()
        self.sampleSize = sampleSize
        self.gridSize = gridSize
        self.conv1 = nn.Conv2d(1, 6, 3, padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=(1, 1))
        self.fc1 = nn.Linear(16 * 5 * 5, outSize)

    #         self.fc2 = nn.Linear(40 + sampleSize * 2, outSize)

    def forward(self, startend, occ):
        occ = self.pool(F.relu(self.conv1(occ)))
        occ = F.relu(self.conv2(occ))
        occ = occ.view(-1, 16 * 5 * 5)
        occ = F.relu(self.fc1(occ))
        x = torch.cat((occ, startend), dim=-1)
        #         x = F.relu(self.fc2(x))
        return x



