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
    dimension: int
        dimension of token embeddings
    seq_length: int
        length of sequence
    mask_ratio: float
        ratio of masked tokens
    patch_size: int
        size of patches
    """
    def __init__(self, dimension, seq_length, mask_ratio, patch_size):
        super().__init__()
        self.k = dimension
        self.seq_length = seq_length
        self.Corruption_ratio = mask_ratio
        self.patch_size = patch_size
        
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
        return unmasked_embeddings, [mask_embedding, unmasked_positions, mask_index, unmask_index]  
    

class MAE_MaskingImage(nn.Module):
    """
    Masking algorithm for the Masked auto encoder model

    - Splits input token embeddings into masked and unmasked tokens
    - creates mask token for masked patches
    - returns masked and unmasked embeddings and positions

    Parameters:
    ------------
    dimension: int
        dimension of token embeddings
    seq_length: int
        length of sequence
    mask_ratio: float
        ratio of masked tokens
    """
    def __init__(self, dimension, seq_length, mask_ratio):
        super().__init__()
        self.k = dimension
        self.seq_length = seq_length
        self.Corruption_ratio = mask_ratio
        
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
        return unmasked_embeddings, [mask_embedding, unmasked_positions, mask_index, unmask_index]

class MAE_Decoder(nn.Module):
    """
    Decoder for the Masked Auto Encoder model

    - takes whole sequence as input after encoder processed unmasked tokens
    - processes masked tokens
    - returns reconstructed sequence

    Parameters:
    ------------
    depth: int
        depth of transformer
    input_dimension: int
        dimension of input tokens
    dimension: int
        dimension of token embeddings
    heads: int
        number of heads in multihead attention
    hidden_dim: int
        hidden dimension of feed forward layer
    """
    def __init__(self, depth, input_dimension, dimension, heads, hidden_dim, dropout):
        super().__init__()
        self.k0 = input_dimension
        self.k = dimension
        depth = depth
        tblocks = []
        for i in range(depth):
            tblocks.append(TBlock(
                depth = depth,
                dimension = self.k,
                heads = heads,
                hidden_dim = hidden_dim,
                dropout = dropout
            ))
        self.tblocks = nn.Sequential(*tblocks)
        self.projection_dec = nn.Linear(self.k, self.k)
        self.act_norm = nn.Sequential(nn.GELU(), nn.LayerNorm(self.k))
        self.pred = nn.Linear(self.k, self.k0)
        
    def forward(self, x):
        output = self.tblocks(x)
        output = self.act_norm(self.projection_dec(output))
        output = self.pred(output)
        return output

class MAE_CreateDecoderInput_Raw(nn.Module):
    """
    For the Masked Auto Encoder model
        -> creates decoder input from encoder output and mask token

    - re-adds positional embeddings to encoder output
    - concats processed encoder outputs with mask tokens in the right order 

    Parameters:
    ------------
    hp: dict
        hyperparameters of model (needs dimension)
    """
    def __init__(self, dimension, sequence_length):
        super().__init__()
        self.k = dimension
        self.t = sequence_length
        self.norm = nn.LayerNorm(self.k)
        
    def forward(self, encoder_output, mask_output):
        """
        Parameters:
        ------------
        encoder_output: torch.tensor
            output of encoder of shape (b, num_not_masked, k)  -> without CLS token!!!
        mask_output: list
            list of mask embeddings, unmasked positions, mask indices and unmasked indices
        """
        mask_embedding, unmasked_positions, mask_id, unmask_id = mask_output
        b, _, _ = encoder_output.size()
        encoder_output = self.norm(encoder_output + unmasked_positions)   # add positional embedding again
        decoder_inputs = torch.zeros((b,self.t,self.k),requires_grad=True).to(encoder_output.device)
        for i in range(b):
            decoder_inputs[i, mask_id[i], :] = mask_embedding[i, torch.arange(mask_id[i].size(0))]
            decoder_inputs[i, unmask_id[i], :] = encoder_output[i, torch.arange(unmask_id[i].size(0))]
        return decoder_inputs

class MAE_CreateDecoderInput_Wavelets(nn.Module):
    """
    For the Masked Auto Encoder model
        -> creates decoder input from encoder output and mask token

    - readds positional embeddings to encoder output
    - concats processed encoder outputs with mask tokens in the right order 

    Parameters:
    ------------
    hp: dict
        hyperparameters of model (needs dimension)
    """
    def __init__(self, dimension, sequence_length):
        super().__init__()
        self.k = dimension
        self.t = sequence_length
        self.norm = nn.LayerNorm(self.k)
        
    def forward(self, encoder_output, mask_output):
        """
        Parameters:
        ------------
        encoder_output: torch.tensor
            output of encoder of shape (b, num_not_masked, k)  -> without CLS token!!!
        mask_output: list
            list of mask embeddings, unmasked positions, mask indices and unmasked indices
        """
        mask_embedding, unmasked_positions, mask_id, unmask_id = mask_output
        b, _, _ = encoder_output.size()
        encoder_output = self.norm(encoder_output + unmasked_positions)   # add positional embedding again
        decoder_inputs = torch.zeros((b,self.t,self.k),requires_grad=True).to(encoder_output.device)
        # create tensors to prevent for loop
        num_corrupted = mask_id.shape[1]
        num_not_corrupted = unmask_id.shape[1]
        batch_indices_masked = torch.arange(b).view(b,1).expand(-1, num_corrupted).to(mask_embedding.device)
        batch_indices_unmasked = torch.arange(b).view(b,1).expand(-1, num_not_corrupted).to(mask_embedding.device)
        # create deconder inputs 
        decoder_inputs[batch_indices_masked, mask_id,:] = mask_embedding
        decoder_inputs[batch_indices_unmasked, unmask_id,:] = encoder_output
        return decoder_inputs
    
class MAE_CalcLoss_Wavelets(nn.Module):
    """
    Calculates loss for Masked Auto Encoder model

    - calculates loss between reconstructed sequence and original sequence
    - unscaled loss for masked tokens
    - adds scaled loss for unmasked tokens (scaled by alpha)

    Parameters:
    ------------
    alpha: float
        scaling factor for unmasked tokens
    batch_size: int
        batch size
    loss_func: nn.Module
        loss function
    """
    def __init__(self, alpha, batch_size, loss_func):
        super().__init__()
        self.alpha = alpha
        self.batch_size = batch_size
        self.loss_func = loss_func
        
    def forward(self, outputs, mask_id, unmask_id, orig_image):
        batch_index = torch.arange(self.batch_size).view(-1, 1).to(outputs.device)
        # get the masked part of output and original input
        masked_outputs = outputs[batch_index,mask_id]
        masked_input = orig_image[batch_index,mask_id]
        # get the unmasked pasrt of output and original input
        unmasked_outputs = outputs[batch_index,unmask_id]
        unmask_images = orig_image[batch_index,unmask_id]
        # loss of masked part
        loss = self.loss_func(masked_outputs, masked_input)
        # loos of unmasked part scaled with factor alpha
        loss_unmasked = self.loss_func(unmasked_outputs, unmask_images)
        # add together
        loss += self.alpha * loss_unmasked
        return loss
    
class MAE_CalcLoss_Raw(nn.Module):
    """Calculates loss for Masked Auto Encoder model

    - calculates loss between reconstructed sequence and original sequence
    - unscaled loss for masked tokens
    - adds scaled loss for unmasked tokens (scaled by alpha)

    loss is normed with batch size to make comparable between different batch sizes

    Parameters:
    ------------
    alpha: float
        scaling factor for unmasked tokens
    batch_size: int
        batch size
    loss_func: nn.Module
        loss function
    """
    def __init__(self, alpha, batch_size, loss_func):
        super().__init__()
        self.alpha = alpha
        self.batch_size = batch_size
        self.loss_func = loss_func
        
    def forward(self, outputs, mask_id, unmask_id, orig_image):
        b, _, _ = outputs.size()
        loss = 0    
        for i in range(b):
            loss += self.loss_func(outputs[i,mask_id[i],:], orig_image[i,mask_id[i],:])/b
            loss += self.alpha * self.loss_func(outputs[i,unmask_id[i],:], orig_image[i,unmask_id[i],:])/b
        return loss