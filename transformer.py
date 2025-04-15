import torch
import torch.nn as nn
import math


##################################
# INITIAL SETUP PROGRESS - Positional Encoding + Multi-Head Self-Attention + (Attention + Feed Forward pending)
##################################

# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding, as introduced in
    "Attention is All You Need" (Vaswani et al., 2017).

    The idea is to add position-dependent signals to word embeddings
    so that the model can incorporate sequence order.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Dimensionality of the embeddings/hidden size.
            max_len: Maximum length of input sequences.
        """
        super(PositionalEncoding, self).__init__()


        # Creating a long enough 'positional' matrix once in log space
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1) 

        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)

        # Applying sin to even indices in the array/cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # The buffer is not a model parameter, but we want it on the same device
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            A tensor of the same shape as x, but with positional encoding added.
        """
        seq_len = x.size(1)
        # Adding the positional encoding to the embedding by broadcasting.
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x



# MULTI-HEAD ATTENTION
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention sublayer.

    Given queries, keys, and values (Q, K, V), it produces an output
    by performing scaled dot-product attention in multiple heads, then
    concatenates the results.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimensionality of the embeddings/hidden size.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # We map the input (embeddings) into Q, K, V by linear transformations
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [batch_size, query_len, d_model]
            key:   [batch_size, key_len, d_model]
            value: [batch_size, key_len, d_model]
            mask:  Optional attention mask, broadcastable to shape
                   [batch_size, n_heads, query_len, key_len].

        Returns:
            - A tensor of shape [batch_size, query_len, d_model].
            - Attention weights (optional for analysis).
        """
        B = query.size(0)

        # Linear projections
        Q = self.w_q(query)  
        K = self.w_k(key)  
        V = self.w_v(value) 

        # Splitting each of Q, K, V into n_heads pieces
        Q = Q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        #Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1)  
        attention = self.dropout(attention)

        # Combining attention back with values
        x = torch.matmul(attention, V) 
        x = x.transpose(1, 2).contiguous() 
        x = x.view(B, -1, self.d_model)    

        # Final linear
        x = self.fc_out(x)  
        return x, attention
    

    

# POSITION-WISE FEED FORWARD
class PositionwiseFeedForward(nn.Module):
    """
    Implements the two-layer feed-forward network used in the Transformer:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Args:
            d_model: Hidden size of the model.
            dim_feedforward: Internal dimension of the feed-forward layer.
            dropout: Dropout probability.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            A tensor of shape [batch_size, seq_len, d_model].
        """
        # Project up -> apply ReLU -> dropout -> project down
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


#UP NEXT (04/15/25): E_LAYER -> ENCODER -> D_LAYER -> DECODER -> TRANSFORMER MODEL