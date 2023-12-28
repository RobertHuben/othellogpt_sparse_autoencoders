import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.tokenizer import encode, decode
from attention_head import MyAttentionHead, MyMultiHeadAttention

class OthelloGPT(nn.Module):

    def __init__(self, num_layers, d_model, n_heads, window_length=64, vocab_size=66, tied_embed=False):
        super().__init__()
        self.num_layers=num_layers
        self.d_model=d_model
        self.n_heads=n_heads
        self.window_length=window_length
        self.vocab_size=vocab_size
        self.d_head=int(d_model/n_heads)
        self.token_embed_table=nn.Embedding(vocab_size, d_model)
        self.position_embed_table=nn.Embedding(window_length, d_model)
        self.unembed=nn.Linear(d_model, vocab_size)

        self.linear_encoders=[nn.Linear(in_features=d_model, out_features=4*d_model, bias=True)]
        self.linear_activation=nn.GELU()
        self.linear_decoders=[nn.Linear(in_features=4*d_model, out_features=d_model, bias=True)]
        self.attention_sublayers=[MyMultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads, use_mask=True) for _ in range(num_layers)]


    def forward(self, input, targets=None):
        '''
        '''
        if input.shape[1]>self.window_length:
            input=input[:,:self.window_length]
        if targets.shape[1]>self.window_length:
            targets=targets[:,:self.window_length]
        positions=torch.arange(self.window_length)
        logits=self.token_embed_table(input)+self.position_embed_table(positions)
        for layer in range(self.num_layers):
            logits=logits+self.attention_sublayers[layer](logits)
            hidden_layer=self.linear_encoders[layer](logits)
            logits=logits+self.linear_decoders[layer](self.linear_activation(hidden_layer))
        logits=self.unembed(logits)
        if targets is None:
            loss=None
        else:
            loss=F.cross_entropy(torch.transpose(logits, dim0=1, dim1=2), targets)
        return logits,loss
    

    def generate(self, input, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss=self(input)
            logits=logits[:,-1,:]
            probs=F.softmax(logits, dim=-1)
            idx_next=torch.multinomial(probs, num_samples=1)
            input=torch.concatenate((input, idx_next), dim=1)
        return input
    

