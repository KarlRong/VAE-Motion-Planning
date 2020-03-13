import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np


class convVAE(nn.Module):
    def __init__(self, sample_size, traj_size, cnnout_size, cond_out_size, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super(convVAE, self).__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        
        self.latent_size = latent_size
        self.condnn = CondNN(sample_size, traj_size, cnnout_size)
        self.encoder = Encoder(sample_size + cond_out_size, encoder_layer_sizes, latent_size)
        self.decoder = Decoder(latent_size +cond_out_size, decoder_layer_sizes, sample_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, startend, traj, occ):
        c, _= self.condnn(startend, traj, occ)
#         print("x size: ", x.shape)
#         print("c size", c.shape)
        mu, logvar = self.encode(torch.cat((x, c), dim=1))
        z = self.reparameterize(mu, logvar)
        return self.decode(torch.cat((z, c), dim=-1)), mu, logvar
    
    def inference(self, startend, traj, occ, num_viz):
        c, alpha = self.condnn(startend, traj, occ)
        z = torch.randn(num_viz, self.latent_size, device = c.device)
        return self.decode(torch.cat((z, c), dim=-1)), alpha
    
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
    def __init__(self, sampleSize, traj_size,  outSize, encoder_dim=64, attention_dim=64):
        super(CondNN, self).__init__()
        self.sampleSize = sampleSize
        self.cnn = Conv3d(outSize)
        self.Attention = Attention(cnn_encode_dim=encoder_dim, condition_dim=sampleSize * 2, attention_dim=attention_dim) # + 3 for position
        self.fc1 = nn.Linear(encoder_dim + sampleSize * 2 + traj_size, outSize)
#         self.fc1 = nn.Linear(encoder_dim + traj_size, outSize)

    def forward(self, startend, traj, occ):
        cnn_encode = self.cnn(occ)
        batch_size = startend.size(0)
        attention_weighted_encoding, alpha = self.Attention(cnn_encode, startend)
        x = torch.cat((attention_weighted_encoding, startend, traj.view(batch_size, -1)), dim=-1)
#         x = torch.cat((attention_weighted_encoding, traj.view(batch_size, -1)), dim=-1)
#         print("condnn cated size:", x.shape)
        x = self.fc1(x)
        return x, alpha

class Conv3d(nn.Module):
    def __init__(self, cnn_out_size):
        super(Conv3d, self).__init__()

        self.adap_pool = nn.AdaptiveAvgPool3d((25, 200, 200))
        self.conv_layer1 = self._make_conv_layer(1, 16)
        self.conv_layer2 = self._make_conv_layer(16, 32)
#         self.conv_layer3 = self._make_conv_layer(64, 124)
        self.conv_layer5=nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=0)
        
        self.adap_pool2 = nn.AdaptiveAvgPool3d((6, 20, 20))

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(2, 3, 3), padding=1),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        x = self.adap_pool(x)
#         print(x.size())
        x = self.conv_layer1(x)
#         print(x.size())
        x = self.conv_layer2(x)
#         print(x.size())
        x=self.conv_layer5(x)
#         print(x.size())
        x = self.adap_pool2(x)
#         print("cnn out size",x.size())

        return x
    

class Attention(nn.Module):
    def __init__(self, cnn_encode_dim, condition_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(cnn_encode_dim + 3, attention_dim) #位置信息
        self.condition_att = nn.Linear(condition_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.position = self.initPosition()
    
    def initPosition(self):
        x = np.linspace(0, 5, 6, dtype='float32')
        y = np.linspace(0, 19, 20, dtype='float32')
        t = np.linspace(0, 19, 20, dtype='float32')
        tv,xv, yv = np.meshgrid(t,x,y)
        xv, yv, tv = xv.reshape((1,-1, 1)), yv.reshape((1,-1, 1)), tv.reshape((1,-1, 1))
        position = torch.from_numpy(np.concatenate((xv, yv, tv), axis = 2))
        
        return position
        
        
    def forward(self, encoder_out, condition):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)      
        encoder_out = encoder_out.permute(0, 2, 3, 4, 1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        self.position = self.position.to(encoder_out.device)
        
        self.position = self.position.expand(batch_size, self.position.shape[1], self.position.shape[2]) #存疑
        encoder_out_pos = torch.cat((encoder_out, self.position), dim = 2)
#         print("cat encoder_out: ", encoder_out_pos.shape)
        att1 = self.encoder_att(encoder_out_pos)
        att2 = self.condition_att(condition)# 依然不清晰
        att2 = att2.unsqueeze(1)
        att2 = att2.expand(batch_size, att1.shape[1], -1)
#         print("att1: ", att1.shape)
#         print("att2: ", att2.shape)
        
#         att = self.relu(self.full_att(att1 + att2))
        att = self.full_att(self.relu(att1 + att2))
#         print("att: ", att.shape)
        
        alpha = self.softmax(att)
#         print("encoder_out: ", encoder_out.shape)
#         print("alpha shape: ", alpha.shape)
        attention_weighted_encoding = (encoder_out * alpha).sum(dim=1)
#         print("attention_weighted_encoding: ", attention_weighted_encoding.shape)
            
        return attention_weighted_encoding, alpha