import torch
from torch import nn
import numpy as np
from .Transformer_Classes import TBlock


class MAE_MaskingRaw(nn.Module):
    """
    Masking algorithm for the Masked auto encoder model

    - Splits input token embeddings into masked and unmasked tokens
    - creates mask token for masked patches
    - returns masked and unmasked embeddings and positions

    Parameters:
    ------------
    hp: dict
        hyperparameters of model (needs dimension and seq_length)
    hp_mask: dict
        hyperparameters of mask algorithm (needs Corruption_ratio)
    """
    def __init__(self, hp, hp_mask):
        super().__init__()
        self.k = hp["dimension"]
        self.seq_length = hp["seq_length"]
        self.Corruption_ratio = hp_mask["Corruption_ratio"]
        self.patch_size = hp["patch_size"]
        
        self.max_masked = int(self.seq_length *self.Corruption_ratio)
        self.max_unmasked = self.seq_length  - self.max_masked
        
        self.mask = nn.Linear(1, self.k, bias=False)        # to create mask token
        self.norm_mask = nn.LayerNorm(self.k)
         
    def forward(self, token_embedding, pe, l):
        """
        Parameters:
        ------------
        token_embedding: torch.tensor
            token embeddings of shape (b, t, k) already with positional embedding
        pe: torch.tensor
            positional embedding of shape (t, k)
        l: torch.tensor
            length of individual time series of shape (b, t)

        Returns:
        ------------
        unmasked_embeddings: torch.tensor
            unmasked embeddings of shape (b, num_not_masked, k) -> input for encoder
        mask_embedding: torch.tensor
            masked embeddings of shape (b, num_masked, k) -> concat this to output of encoder -> input for decoder
        unmasked_positions: torch.tensor
            positional embeddings of unmasked tokens of shape (b, num_not_masked, k) -> add this to output of encoder
        mask_index: list
            list of indices of masked tokens -> needed for loss calculation
        unmask_index: list
            list of indices of unmasked tokens -> needed for loss calculation
        """
        b, t, k0 = token_embedding.size()
        mask_index = []
        unmask_index = []
        mask = torch.tensor([1.],requires_grad=True).to(token_embedding.device)
        mask = self.mask(mask)      # mask token shape (embed_dimention k)
        unmasked_embeddings = torch.zeros(b, self.max_unmasked, self.k).to(token_embedding.device)
        unmasked_positions = torch.zeros(b, self.max_unmasked, self.k).to(token_embedding.device)
        mask_embedding = torch.zeros(b, self.max_masked, self.k).to(token_embedding.device)
        for i in range(b):
            ind_length = int(l[i]/self.patch_size)      # length of individual time series divided by patch size
            num_masked = int(self.Corruption_ratio*ind_length) 
            num_not_masked = ind_length - num_masked
            rand_index = torch.randperm(ind_length)
            mask_index.append(torch.sort(rand_index[:num_masked])[0].int())
            unmask_index.append(torch.sort(rand_index[num_masked:])[0].int())
            mask_embedding[i,torch.arange(num_masked)] = self.norm_mask(mask[None, :].expand(num_masked, self.k) + pe[mask_index[i]])
            unmasked_embeddings[i,torch.arange(num_not_masked)] = token_embedding[i, unmask_index[i]]
            unmasked_positions[i,torch.arange(num_not_masked)] = pe[unmask_index[i]]
        return unmasked_embeddings, mask_embedding, unmasked_positions, mask_index, unmask_index  
    

class MAE_MaskingImage(nn.Module):
    """
    Masking algorithm for the Masked auto encoder model

    - Splits input token embeddings into masked and unmasked tokens
    - creates mask token for masked patches
    - returns masked and unmasked embeddings and positions

    Parameters:
    ------------
    hp: dict
        hyperparameters of model (needs dimension and seq_length)
    hp_mask: dict
        hyperparameters of mask algorithm (needs Corruption_ratio)
    """
    def __init__(self, hp, hp_mask):
        super().__init__()
        self.k = hp["dimension"]
        self.seq_length = hp["seq_length"]
        self.Corruption_ratio = hp_mask["Corruption_ratio"]
        self.patch_size = hp["patch_size"]
        
        self.num_masked = int(self.seq_length * self.Corruption_ratio)
        self.num_unmasked = self.seq_length  - self.num_masked
        
        self.mask = nn.Linear(1, self.k, bias=False)        # to create mask token
        self.norm_mask = nn.LayerNorm(self.k)
         
    def forward(self, token_embedding, pe):
        """
        Parameters:
        ------------
        token_embedding: torch.tensor
            token embeddings of shape (b, t, k) already with positional embedding
        pe: torch.tensor
            positional embedding of shape (t, k)

        Returns:
        ------------
        unmasked_embeddings: torch.tensor
            unmasked embeddings of shape (b, num_not_masked, k) -> input for encoder
        mask_embedding: torch.tensor
            masked embeddings of shape (b, num_masked, k) -> concat this to output of encoder -> input for decoder
        unmasked_positions: torch.tensor
            positional embeddings of unmasked tokens of shape (b, num_not_masked, k) -> add this to output of encoder
        mask_index: torch tensor
            list of indices of masked tokens -> needed for loss calculation
        unmask_index: torch tensor
            list of indices of unmasked tokens -> needed for loss calculation
        """
        b, t, k0 = token_embedding.size()
        mask_index = torch.empty((b, self.num_masked)).to(token_embedding.device) 
        unmask_index = torch.empty((b, self.num_unmasked)).to(token_embedding.device) 
        for i in range(b):
            rand_idx = torch.randperm(t)
            mask_index[i] = torch.sort(rand_idx[:self.num_masked])[0]
            unmask_index[i] = torch.sort(rand_idx[self.num_masked:])[0]
        mask_index = mask_index.int()
        unmask_index = unmask_index.int()
        #create mask token
        mask = torch.tensor([1.],requires_grad=True).to(token_embedding.device)
        mask = self.mask(mask)      # mask token shape (embed_dimention k)
        
        # created tensors to prevent for loop
        batch_indices_masked = torch.arange(b).view(b,1).expand(-1, self.num_masked).to(token_embedding.device)
        batch_indices_unmasked = torch.arange(b).view(b,1).expand(-1, self.num_unmasked).to(token_embedding.device)
        # create seperate tensors with unmasked embeddings and unmasked positions (containing position embedding of the unmasked tokens)
        unmasked_embeddings = token_embedding[batch_indices_unmasked, unmask_index]
        # expand positional embedding for every batch
        pe = pe[None, :,:].expand(b,self.seq_length,self.k)
        unmasked_positions = pe[batch_indices_unmasked, unmask_index]
        # create mask_embedding -> containing mask token for every masked position and add position embedding
        mask_embedding = self.norm_mask(mask[None, None, :].expand(b, self.num_masked, self.k) + pe[batch_indices_masked, mask_index])
        return unmasked_embeddings, mask_embedding, unmasked_positions, mask_index, unmask_index

class MAE_Decoder(nn.Module):
    """
    Decoder for the Masked Auto Encoder model

    - takes whole sequence as input after encoder processed unmasked tokens
    - processes masked tokens
    - returns reconstructed sequence

    Parameters:
    ------------
    hp: dict
        hyperparameters of model (needs input_dimension, dimension and depth_decoder)
    """
    def __init__(self, hp):
        super().__init__()
        self.k0 = hp["input_dimension"]
        self.k = hp["dimension"]
        depth = hp["depth_decoder"]
        tblocks = []
        for i in range(depth):
            tblocks.append(TBlock(hp))
        self.tblocks = nn.Sequential(*tblocks)
        self.projection_dec = nn.Linear(self.k, self.k)
        self.act_norm = nn.Sequential(nn.GELU(), nn.LayerNorm(self.k))
        self.pred = nn.Linear(self.k, self.k0)
        
    def forward(self, x):
        output = self.tblocks(x)
        output = self.act_norm(self.projection_dec(output))
        output = self.pred(output)
        return output