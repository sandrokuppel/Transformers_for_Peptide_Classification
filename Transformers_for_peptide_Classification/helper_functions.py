import torch
from torch import nn

# creates positional encodings for a sequence of length max_sequence_length and dimension d_model
# positional encoding like in paper "Attention is all you need"
class Positionalencoding(nn.Module):
    """
    Positional Encoding Class
    
    Created position encoding like propoesed in the paper "Attention is all you need"
    """
    def __init__(self, max_sequence_length, d_model):
        """
        Initializes the Positional Encoding class
        
        Parameters:
        ------------
        d_model : int
            model dimension
        max_sequence_length : int
            maximum sequence length
        """
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model 
        
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    
class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lr, epochs, steps_per_epoch, pct_start, div_factor, final_div_factor, last_epoch=-1):
        self.total_steps = int(epochs*steps_per_epoch)
        self.warmup_steps = int(self.total_steps*pct_start)
        self.init_lr = max_lr/div_factor
        self.final_lr = max_lr/final_div_factor
        self.max_lr = max_lr
        self.lr_gab_warmup = max_lr-self.init_lr
        self.lr_gab_decay = max_lr-self.final_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [self.init_lr + self.lr_gab_warmup * step / self.warmup_steps]
        else:
            return [self.max_lr - self.lr_gab_decay * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)]


def rename_keys(state_dict, rename_func, prefix):
    return {rename_func(prefix, key): value for key, value in state_dict.items()}
def prepend_prefix(prefix,key):
    return prefix + key
def remove_prefix(prefix,key):
    return key.replace(prefix, '', 1)
def remove_keys(state_dict, keys_to_remove):
    return {key: value for key, value in state_dict.items() if key not in keys_to_remove}