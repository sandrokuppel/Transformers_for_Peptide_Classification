import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .Transformer_Classes import TBlock


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
    
class CrossTBlock_TimeCatch(nn.Module):
    def __init__(self, hp, depth, cross_att_freq):
        super().__init__()
        """
        Class to use Transformer architecture for Raw data 
        combined with feed forward for catch22
        and cross attention CLS with catch22
        -> only CLS gets infromation from catch22

        Architecture:
            - all attention for time series branch
            - cross attention time series cls in catch22

        Parameters:
        -----------
        hp: dict
            hyperparameters
        depth: int
            depth of this particular class
        """
        dim_list = hp['dimension_list']
        heads_list = hp['heads']
        dropout = hp['dropout']
        self.channels = hp['channels']
        self.cross_att_freq = cross_att_freq
        self.depth = depth
        self.tblock = TBlock(hp, branch=0)
        self.CA = CrossAttention(dim_list[0], dim_list[1], heads_list[0], dropout)
        
    def forward(self, x):   # x is list of different branches 
        catch25 = x[1]
        emb_time_series = self.tblock(x[0])
        cls_time_series = emb_time_series[:,0:1,:]
        if self.depth % self.cross_att_freq == 0:
            print(self.depth)
            cls_time_series = self.CA(cls_time_series, catch25)
        emb_time_series = torch.cat((cls_time_series, emb_time_series[:,1:,:]), dim=1)
        return [emb_time_series, catch25]
    
class TimeCatch_Encoder(nn.Module):
    """
    Encoder for Time Series data with Catch22 data
    """
    def __init__(self,hp):
        super().__init__()
        depth = hp["depth_encoder"]
        cross_att_freq = hp["cross_att_freq"]
        
        tblocks = []
        for i in range(depth):
            tblocks.append(CrossTBlock_TimeCatch(hp, i, cross_att_freq=cross_att_freq))
        self.tblocks = nn.Sequential(*tblocks)
        
    def forward(self, x):
        emb = self.tblocks(x)
        return emb
    

class CrossViT_Encoder(nn.Module):
    """
    Encoder Class for CrossViT architecture
    """
    def __init__(self,hp):
        super().__init__()
        depth = hp["depth"]
        cross_att_freq = hp["cross_att_freq"]
        
        tblocks = []
        for i in range(depth):
            tblocks.append(CrossTBlock(hp, i, cross_att_freq=cross_att_freq))
        self.tblocks = nn.Sequential(*tblocks)
        
    def forward(self, x):
        emb = self.tblocks(x)
        return emb
    
class CrossViT_Decoder(nn.Module):
    '''
    Decoder Class for multi branch architecture
    '''
    def __init__(self, hp):
        super().__init__()
        depth = hp["depth_decoder"]
        self.k = hp["dimension_list"]
        self.channels = hp["channels"]
        self.k0 = hp["input_dimension_list"]
        
        tblocks = []
        for i in range(depth):
            tblocks.append(CrossTBlock(hp, i , 0))      # no cross attention in decoder
        self.tblocks = nn.Sequential(*tblocks)
        self.act_norm = nn.ModuleList([nn.Sequential(nn.Linear(self.k[i], self.k[i]), nn.GELU(), nn.LayerNorm(self.k[i])) for i in range(self.channels)])
        self.pred = nn.ModuleList([nn.Linear(self.k[i], self.k0[i]) for i in range(self.channels)])
        
    def forward(self, x):
        emb = self.tblocks(x)
        output = [self.act_norm[i]((emb[i])) for i in range(self.channels)]     # remove CLS token
        output = [self.pred[i](output[i]) for i in range(self.channels)]
        return output
    
class CrossTBlock(nn.Module):
    def __init__(self, hp, depth, cross_att_freq):
        """
        Transformer Block for multiple branches
        Automatically does the cross attention for all the branches with the given frequency
            -> needs to get the depth when initialized to check if cross attention should be done

        Parameters:
        -----------
        hp: dict
            hyperparameters
        depth: int
            position of this Block 
        """
        super().__init__()
        dim_list = hp['dimension_list']
        heads_list = hp['heads_list']
        hidden_dim_list = hp['hidden_dim_list']
        dropout = hp['dropout']
        self.channels = hp['channels']      # number of channels
        self.cross_att_freq = cross_att_freq
        self.depth = depth
        self.tblocks = nn.ModuleList([TBlock(hp['depth'], dim_list[i], heads_list[i], hidden_dim_list[i], hp['dropout']) for i in range(self.channels)])
        CA_List = []
        if self.cross_att_freq != 0 and self.depth % self.cross_att_freq == 0:
                for i in range(self.channels):
                    CA_List.append(nn.ModuleList([CrossAttention(dim_list[i], dim_list[j], heads_list[j], dropout) for j in range(self.channels) if j != i]))
        self.CA_List = nn.ModuleList(CA_List)
        
    def forward(self, x):   # x is list of different branches 
        emb = [self.tblocks[i](x[i]) for i in range(self.channels)]                  # all attention on branch i
        cls = [emb[i][:,0:1,:] for i in range(self.channels)]                        # extract cls tokens
        if self.cross_att_freq != 0 and self.depth % self.cross_att_freq == 0:       # do cross attention with given frequency for all the branches
            for i in range(self.channels):
                idx = 0
                for k in range(self.channels):
                    if i != k:
                        cls[i] = self.CA_List[i][idx](cls[i], emb[k][:,1:,:])         # cross attention cls[i] in branch k dont use cls token of branch k
                        idx += 1
                    
        for i in range(self.channels):
            emb[i] = torch.cat((cls[i], emb[i][:,1:,:]), dim=1)                       # replace old cls token
        return emb
    
