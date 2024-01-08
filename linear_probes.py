import torch
import torch.nn as nn
from torch.nn import functional as F
# from utils.tokenizer import encode, decode

class LinearProbe(torch.nn.Module):

    def __init__(self, othello_gpt_model, layer_num):
        super().__init__()
        self.othello_gpt_model=othello_gpt_model
        self.layer_num=layer_num
        self.board_size=64
        self.classifier=nn.ModuleList(nn.Linear(othello_gpt_model.d_model, self.board_size) for _ in range(3))
        # freeze othello gpt model weights
        for parameter in self.othello_gpt_model.parameters():
            parameter.requires_grad=False

    def forward(self, input, target=None):
        #input is B-by-W, where W is the length of the context window
        #target are B-W-64, and are 0,1,2 for the 3 classes, with -100 for indices that should be ignored (e.g after the game)
        logits=self.othello_gpt_model.intermediate_residual_stream(input, self.layer_num) #B-W-512
        predictions=torch.stack([classifier_shard(logits) for classifier_shard in self.classifier], dim=3) #B-W-64-3
        if target is None:
            loss=None
        else:
            loss_function=torch.nn.CrossEntropyLoss()
            loss=loss_function(input=predictions.flatten(end_dim=-2), target=target.flatten())
        return predictions, loss
