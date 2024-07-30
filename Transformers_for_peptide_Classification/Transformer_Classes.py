import torch
from torch import nn
import numpy as np
from .helper_functions import Positionalencoding

class TBlock(nn.Module):
    """
    Transformer Block Class
    
    This is the main building block of the transformer model. 
    -> multihead attention
    -> layer normalization
    -> feedforward neural network
    -> layer normalization 
    Has the option to return attention weights
    
    Parameters:
    ------------
    hp : dict
        hyperparameters of the model
    block_number : int
        block number of the transformer block (needed for getting attention weights)
    """
    def __init__(self, hp, block_number = None):
        super().__init__()
        heads = hp["heads"]
        k = hp["dimension"]
        hidden_dim = hp["hidden_dim"]
        dropout = hp["dropout"]
        self.depth = hp['depth']
        self.block_number=block_number

        self.attention = nn.MultiheadAttention(k, heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(k, hidden_dim),      
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(hidden_dim, k))
        
    def forward(self, x, get_weights=False):
        """
        Forward pass of the transformer block
        
        Parameters:
        ------------
        x : torch.Tensor
            input tensor
        get_weights : bool
            whether to return attention weights or not
        
        Returns:
        ------------
        torch.Tensor
            output tensor of the transformer block or attention weights
        """
        attended, weights = self.attention(x, x, x, need_weights=True,  average_attn_weights = False)
        if get_weights and self.block_number == self.depth-1:
            return weights
        attended = self.dropout1(attended)
        x = self.norm1(attended + x)
        feedforward = self.dropout2(self.ff(x))
        return self.norm2(feedforward + x)
    
class TPrep(nn.Module):
    """
    Transformer prepare embeddings class

    Takes input of shape (b, t, k0): (batch, sequence, input dimension)
    outputs CLS: classification token, token_embedding: embedded sequence (b,t,k), positional embedding: pe
    If model is pretrained, the embeddings are frozen -> only CLS token is learned

    Parameters:
    ------------
    hp: dict 
        hyperparameters
    learnable_pos_embedding: bool
        True -> learned pos embedding
        False -> pos encoding as in attention is all you need paper
    """
    def __init__(self, hp, learnable_pos_emb=True, cls_token=True):
        super().__init__()
        self.k0 = hp["patch_size"]      #input dimension
        self.k = hp["dimension"]
        self.seq_length = hp["seq_length"]
        Pretrained = hp["Pretraining"]
        self.learnable_emb = learnable_pos_emb
        self.cls_token = cls_token

        if cls_token:
            self.CLS = nn.Linear(1, self.k, bias=False)
        self.embed_tokens = nn.Sequential(nn.Linear(self.k0, self.k),
                                          nn.LayerNorm(self.k))
        if self.learnable_emb:
            self.pos_embedding = nn.Embedding(self.seq_length, self.k)    
        else: 
            self.pos_embedding = Positionalencoding(self.seq_length, self.k)  
        
        if Pretrained:
            for param in self.pos_embedding.parameters():
                param.requires_grad = False
            for param in self.embed_tokens.parameters():
                param.requires_grad = False

        self.layer_norm = nn.LayerNorm(self.k)

    def forward(self, x):
        b, t, k0 = x.size()
        assert (k0 == self.k0), 'dimension does not match'
        assert (t == self.seq_length),   'sequence length does not match'
        token_embedding = self.embed_tokens(x)                               # go to dimension k
        if self.cls_token:
            CLS = torch.tensor([1.],requires_grad=True).to(x.device)             # CLS token in shape (1,1
            CLS = self.CLS(CLS)[None, :].expand(b,self.k)[:,None,:].to(x.device)     #CLS token in shape (b,1,k)
        if self.learnable_emb:
            pe_out = self.pos_embedding.weight
            pe = pe_out[None, :,:].expand(b,self.seq_length,self.k).to(x.device) # expand to create for every batch 
        else:
            pe = self.pos_embedding()[None, :,:].expand(b,self.seq_length,self.k).to(x.device)
        token_embedding = self.layer_norm(token_embedding + pe)
        if self.cls_token:
            return CLS, token_embedding, pe_out
        else:
            return token_embedding, pe_out

class MaskAlgorythmRaw(nn.Module):
    """
    Masking Algorythm Class for raw time series data

    Takes sequence (wihtout CLS) as input,
    masks some tokens (either with [mask], [random] or no masking)
    Difference to image mask algorythgm:
        needs length of the individual time series 

    returns:
        x: masked sequence
        I: index of masked tokens (for loss calculation)
        mask_type: 0,1,2 -> [mask], random, no masking
    """
    def __init__(self, hp_mask, hp):
        super().__init__()
        self.P_m = hp_mask["P_m"]                           # probability of mask token
        self.P_r = hp_mask["P_r"]                           # probability of random token 
        self.P_e = hp_mask["P_e"]                           # probabilitz of equal token
        self.Corruption_ratio = hp_mask["Corruption_ratio"] # ratio of corrupted tokens
        
        self.k = hp["dimension"]
        self.seq_length = hp["seq_length"]
        self.batch_size = hp["batch_size"]
        self.patch_size = hp["patch_size"]

        self.num_mask = int(self.batch_size*self.P_m)       # number of mask pictures in one batch
        self.num_rand = int(self.batch_size*self.P_r)       # number of pictures with random tokens in one batch
        self.num_equal = int(self.batch_size*self.P_e)      # number of pictures with equal tokens in one batch
        self.mask_type = torch.zeros((self.batch_size))

        self.num_corrupted = int(self.Corruption_ratio*self.seq_length) # number of corrupted tokens in one picture 

        self.mask = nn.Linear(1, self.k, bias=False)        # to create mask token

    def forward(self, x, len, pos):           # input shape: (b, t, k)
        mask = torch.tensor([1.],requires_grad=True).to(x.device)
        mask = self.mask(mask)      # mask token shape (sequence length)
        rand_ind = torch.from_numpy(np.random.choice(range(self.batch_size), self.batch_size, replace=False))  # random shuffled indices over batch size
        mask_b = rand_ind[:self.num_mask]                                             # extract indices for mask pictures
        rand_b = rand_ind[self.num_mask:self.num_mask+self.num_rand]                  # extract indices for random pictures
        equal_b = rand_ind[self.num_mask+self.num_rand:] # extract indices for equal pictures  
        self.mask_type[mask_b] = 0
        self.mask_type[rand_b] = 1
        self.mask_type[equal_b] = 2
        I=[]
        for i in range(self.batch_size):
            self.num_corrupted = int((self.Corruption_ratio*len[i]/self.patch_size))
            mask_id = torch.from_numpy(np.random.choice(range(int(len[i]/self.patch_size)), self.num_corrupted, replace=False)).to(x.device)
            if self.mask_type[0] == 0:
                x[i, mask_id] = mask + pos[mask_id]             # plus one cause of CLS token
            elif self.mask_type[0] == 1:
                x[i, mask_id] = torch.rand(self.num_corrupted,self.k).to(x.device) + pos[mask_id]
            I.append(mask_id)
        return x, I, self.mask_type


class Encoder(nn.Module):
    """
    Encoder of Transformer

    Puts together the Tranformer blocks for the encoding

    Forward has the option to return the attention weights of the las transformer block
    """
    def __init__(self,hp):
        super().__init__()
        self.depth = hp['depth']
        self.tblocks= nn.ModuleList([TBlock(hp, block_number=i)  for i in range(self.depth)])

    def forward(self, x, len=None, get_weights=False):
        for i in range(self.depth):
            output = self.tblocks[i](x, get_weights)
        return output

class ClassifierCLS(nn.Module):
    """
    Classifier for Classification token

    Takes CLS as input and outputs tensor with size of number of classes
    No softmax applied
    """
    def __init__(self, hp):
        super().__init__()
        self.k = hp["dimension"]
        num_classes = hp["num_classes"]

        self.linear1 = nn.Linear(self.k, int(self.k/2))
        self.GELU = nn.GELU()
        self.linear2 = nn.Linear(int(self.k/2), num_classes)
    def forward(self, x):
        x = self.GELU(self.linear1(x))
        output = self.linear2(x)
        return output
    

class ViTDecoder(nn.Module):
    """
    Decoder for after pretraining like in  Vision Transformer
    also like in original paper
    
    Projects the output of the encoder to the original input dimension
    -> tries to recreate original patches 

    Parameters:
    ------------
    hp: dict
        hyperparameters of model (needs input and output dimension)
    """
    def __init__(self, hp):
        super().__init__()
        self.k = hp["dimension"]
        self.k0 = hp["patch_size"]

        self.projection_dec = nn.Linear(self.k, self.k)
        self.act_norm = nn.Sequential(nn.GELU(), nn.LayerNorm(self.k))
        self.pred = nn.Linear(self.k, self.k0)

    def forward(self, x):
        x = self.projection_dec(x)
        x = self.act_norm(x)
        x = self.pred(x)
        return x