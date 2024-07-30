import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, cls_dim, emb_dim,  num_heads=4, dropout=0.):
        super().__init__()
        """
        Implementation of Cross Attention Module

        Performs cross attention of given cls token with the given sequence
        add and norm the cls token with the output of the cross attention
            -> once in emb dime and again in cls dim 

        Parameters
        ----------
        cls_dim : int
            Dimension of the cls token
        emb_dim : int
            Dimension of the input sequence
        num_heads : int, optional
            Number of heads in the multihead attention, by default 4
        dropout : float, optional
            Dropout probability, by default 0.
        """
        self.num_heads = num_heads
        head_dim = emb_dim // num_heads
        self.scale = head_dim ** -0.5
        qkv_bias = False
        attn_drop = dropout
        proj_drop = dropout
        self.wq = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.forward_transform = nn.Linear(cls_dim, emb_dim, bias=None)         # convert cls to embed dimension 
        self.inverse_transform = nn.Linear(emb_dim, cls_dim, bias=None)         # convert cls to back to own branch dimension 
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(cls_dim)
        
    def forward(self, cls, x):
        b, t, k = x.shape
        cls_ = self.norm1(self.forward_transform(cls))
        
        query = self.wq(cls_).reshape(b, 1, self.num_heads, k // self.num_heads).permute(0, 2, 1, 3)  # b1k -> b1h(k/h) -> bh1(k/h)
        key = self.wk(x).reshape(b, t, self.num_heads, k // self.num_heads).permute(0, 2, 1, 3)  # btk -> bth(k/h) -> bht(k/h)
        value = self.wv(x).reshape(b, t, self.num_heads, k // self.num_heads).permute(0, 2, 1, 3)  # btk -> bth(k/h) -> bht(k/h)

        attn = (query @ key.transpose(-2, -1)) * self.scale  # bh1(k/h) @ bh(k/h)t -> bh1t
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(b, 1, k)   # (bh1t @ bht(k/h)) -> bh1(k/h) -> b1h(k/h) -> b1k
        x = self.proj(x)
        x = self.proj_drop(x)
        cls_ = self.norm2(cls_ + x)
        cls_new = self.norm3(cls + self.inverse_transform(cls_))
        return cls_new