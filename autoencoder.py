import torch
import torch.nn as nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):

    def __init__(self, input_size, feature_ratio, sparsity_coeff=.1):
        super().__init__()
        self.input_size=input_size
        self.sparsity_coeff=sparsity_coeff
        self.hidden_layer_size=int(input_size*feature_ratio)
        self.encoder=nn.Linear(self.input_size, self.hidden_layer_size)
        self.decoder=nn.Linear(self.hidden_layer_size, self.input_size, bias=False)
        self.decoder.weight[:] = self.encoder.weight.T[:] #tied weights
        self.activation=F.relu()


    def forward(self, input):
        hidden_layer=self.activation(self.encoder(input))
        reconstruction=self.decoder(hidden_layer)
        sparsity_loss=torch.norm(hidden_layer, p=1)
        reconstruction_loss=torch.norm(input-reconstruction, p=2)
        total_loss=reconstruction_loss+self.sparsity_coeff*sparsity_loss
        return reconstruction, hidden_layer, total_loss


