import torch
from torch import nn

class TBlock(nn.Module):
    def __init__(self, hp, Pretraining=False):
        super().__init__()
        heads = hp["heads"]
        k = hp["dimension"]
        hidden_dim = hp["hidden_dim"]
        batch_size = hp["batch_size"]
        seq_length = hp["seq_length"]
        dropout = hp["dropout"]

        self.key_padding_mask = None
        self.Pretraining = Pretraining

        if Pretraining:
            self.key_padding_mask = torch.zeros((batch_size, seq_length+1), dtype=torch.bool)
            self.key_padding_mask[:,0] = True
        self.attention = nn.MultiheadAttention(k, heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(k, hidden_dim),      
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(hidden_dim, k))
        
    def forward(self, x):
        if self.Pretraining:
            self.key_padding_mask = self.key_padding_mask.to(x.device)
        attended = self.dropout1(self.attention(x, x, x, need_weights=False, key_padding_mask=self.key_padding_mask)[0])
        x = self.norm1(attended + x)
        feedforward = self.dropout2(self.ff(x))
        return self.norm2(feedforward + x)