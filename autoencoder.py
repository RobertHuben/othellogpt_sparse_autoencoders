import torch
import torch.nn as nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):

    def __init__(self, input_size, feature_ratio, sparsity_coeff=.1, initialization_weights=None):
        super().__init__()
        self.input_size=input_size
        self.sparsity_coeff=sparsity_coeff
        self.hidden_layer_size=int(input_size*feature_ratio)
        if initialization_weights==None:
            initialization_weights=torch.normal(0, 1, (self.input_size, self.hidden_layer_size))
        self.encoder_decoder_matrix=nn.Parameter(initialization_weights)
        self.encoder_bias=nn.Parameter(torch.normal(0,1, (self.hidden_layer_size,)))
        self.activation=torch.nn.ReLU()
        # freeze othello gpt model weights
        for parameter in self.othello_gpt_model.parameters():
            parameter.requires_grad=False


    def forward(self, input):
        normalized_matrix=F.normalize(self.encoder_decoder_matrix, p=2, dim=1)
        hidden_layer=self.activation(input@normalized_matrix + self.encoder_bias)
        reconstruction=hidden_layer@normalized_matrix.transpose(0,1)
        sparsity_loss=torch.norm(hidden_layer, p=1)/hidden_layer.numel()
        reconstruction_loss=torch.norm(input-reconstruction, p=2)/input.numel()
        total_loss=reconstruction_loss+self.sparsity_coeff*sparsity_loss
        return reconstruction, hidden_layer, total_loss, reconstruction_loss, sparsity_loss


