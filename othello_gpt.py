import torch
import torch.nn as nn
from torch.nn import functional as F
# from utils.tokenizer import encode, decode

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

        self.linear_activation=nn.GELU()
        self.mlps=[myMLPLayer(d_model=self.d_model, activation=self.linear_activation) for _ in range(num_layers)]
        # self.linear_encoders=[nn.Linear(in_features=d_model, out_features=4*d_model, bias=True) for _ in range(num_layers)]
        # self.linear_decoders=[nn.Linear(in_features=4*d_model, out_features=d_model, bias=True) for _ in range(num_layers)]
        self.attention_sublayers=[MyMultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads, use_mask=True) for _ in range(num_layers)]


    def forward(self, input, targets=None):
        '''
        '''
        if input.shape[1]>self.window_length:
            input=input[:,:self.window_length]
        if targets != None and targets.shape[1]>self.window_length:
            targets=targets[:,:self.window_length]
        positions=torch.arange(self.window_length)
        logits=self.token_embed_table(input)+self.position_embed_table(positions)
        for layer in range(self.num_layers):
            logits=logits+self.attention_sublayers[layer](logits)
            logits=logits+self.mlps[layer](logits)
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
    


class MyAttentionHead(torch.nn.Module):

    def __init__(self, d_head, d_model, use_mask=True):
        super().__init__()
        self.d_model=d_model
        self.d_head=d_head
        self.scaling_factor=1/torch.sqrt(torch.tensor([d_model]))
        self.Q = torch.nn.Linear(d_model, d_head, bias=False) 
        self.K = torch.nn.Linear(d_model, d_head, bias=False)
        self.V = torch.nn.Linear(d_model, d_head, bias=False)
        # self.O = torch.nn.Linear(d_head, d_model, bias=False)
        self.use_mask=use_mask

    def forward(self, residual_stream):
        attention=self.attention_pattern(residual_stream)
        x=self.V(residual_stream)
        return attention@x

    def attention_pattern(self, residual_stream):
        keys=self.K(residual_stream)
        queries=self.Q(residual_stream)
        pre_attention=queries@torch.transpose(keys, dim0=1, dim1=2)*self.scaling_factor
        if self.use_mask:
            upper_triangular=torch.tril(torch.ones(pre_attention.shape))
            pre_attention=pre_attention.masked_fill(upper_triangular==0, float('-inf'))
        attention=F.softmax(pre_attention, dim=-1)
        return attention
        

class MyMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, use_mask=True):
        super().__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_head=d_model//n_heads
        self.heads=nn.ModuleList([MyAttentionHead(d_head=self.d_head, d_model=d_model, use_mask=use_mask) for _  in range(n_heads)])
        self.proj=nn.Linear(self.n_heads*self.d_head, d_model)

    def forward(self, residual_stream):
        all_heads_output=torch.cat([head(residual_stream) for head in self.heads], dim=-1)
        output=self.proj(all_heads_output)
        return output

class myMLPLayer(torch.nn.Module):

    def __init__(self, d_model, activation):
        super().__init__()
        self.encode=nn.Linear(in_features=d_model, out_features=4*d_model, bias=True)
        self.proj=nn.Linear(in_features=4*d_model, out_features=d_model, bias=True)
        self.activation=activation

    def forward(self, residual_stream):
        hidden_layer=self.activation(self.encode(residual_stream))
        output=self.proj(hidden_layer)
        return output


class MyTransformerBlock(torch.nn.Module):
    pass