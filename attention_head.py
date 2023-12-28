import torch
from torch.nn import functional as F
import math
class MyAttentionHead(torch.nn.Module):

    def __init__(self, d_head, d_model, use_mask=True):
        super().__init__()
        self.d_model=d_model
        self.d_head=d_head
        scaling_factor=1/math.sqrt(d_model)
        self.Q = torch.nn.Parameter(scaling_factor*torch.randn((d_model, d_head)))
        self.K = torch.nn.Parameter(scaling_factor*torch.randn((d_model, d_head)))
        self.O = torch.nn.Parameter(scaling_factor*torch.randn((d_model, d_head)))
        self.V = torch.nn.Parameter(scaling_factor*torch.randn((d_model, d_head)))
        self.use_mask=use_mask

    def forward(self, residual_stream):
        attention=self.attention_pattern(residual_stream)
        x=(residual_stream@self.V)@torch.transpose(self.O, dim0=0, dim1=1)
        return attention@x

    def attention_pattern(self, residual_stream):
        keys=residual_stream@self.K
        queries=residual_stream@self.Q
        pre_attention=queries@torch.transpose(keys, dim0=1, dim1=2)
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
        self.heads=[MyAttentionHead(d_head=self.d_head, d_model=d_model, use_mask=use_mask) for _  in range(n_heads)]

    def forward(self, residual_stream):
        all_heads_output=torch.stack([head.forward(residual_stream) for head in self.heads])

        output=torch.sum(all_heads_output, dim=0)
        return output
