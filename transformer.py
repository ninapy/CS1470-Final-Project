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


# Encoder Layer
class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer, made up of:
      - Multi-Head Self-Attention (with residual & layer norm)
      - Position-wise Feed Forward (with residual & layer norm)
    """
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        """
        Args:
            src: [batch_size, src_len, d_model]
            src_mask: [batch_size, 1, src_len, src_len] (optional)
        Returns:
            Encoded output of shape [batch_size, src_len, d_model].
        """
        # Self-attention sub-layer
        attn_output, _ = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))

        # Position-wise feed-forward sub-layer
        ff_output = self.ff(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src


# Encoder
class Encoder(nn.Module):
    """
    Transformer Encoder that stacks multiple EncoderLayer layers.
    """
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 dim_feedforward: int,
                 dropout: float,
                 max_len: int = 5000):
        super(Encoder, self).__init__()
        self.d_model = d_model

        # Token embedding
        self.embed_tokens = nn.Embedding(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Dropout after embedding
        self.dropout = nn.Dropout(dropout)

        # Stack of EncoderLayers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        """
        Args:
            src: [batch_size, src_len]  (token indices)
            src_mask: [batch_size, 1, src_len, src_len]
        Returns:
            Encoded representations: [batch_size, src_len, d_model]
        """
        #Embed tokens + scale
        src_embed = self.embed_tokens(src) * math.sqrt(self.d_model)
        #Add positional encoding
        src_embed = self.pos_encoding(src_embed)
        #Apply dropout
        src_embed = self.dropout(src_embed)

        # Pass through each layer
        out = src_embed
        for layer in self.layers:
            out = layer(out, src_mask)

        return out
    


# Decoder Layer
class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer, made up of:
      - Masked Multi-Head Self-Attention (with residual & layer norm)
      - Multi-Head Attention over Encoder output (with residual & layer norm)
      - Position-wise Feed Forward (with residual & layer norm)
    """
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super(DecoderLayer, self).__init__()
        # self-attention for the target tokens
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # cross-attention: query is the decoder, key/value is the encoder output
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor,
                tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None):
        """
        Args:
            tgt: [batch_size, tgt_len, d_model]
            enc_out: [batch_size, src_len, d_model]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
            src_mask: [batch_size, 1, 1, src_len]
        Returns:
            (out, attn_weights) 
            out: shape [batch_size, tgt_len, d_model]
            attn_weights: optional, for analyzing cross-attention
        """
        #Masked self-attention
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        x = self.norm1(tgt + self.dropout(self_attn_out))

        #Encoder-Decoder cross-attention
        cross_attn_out, attn_weights = self.cross_attn(x, enc_out, enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        #Position-wise feed forward
        ff_out = self.ff(x)
        out = self.norm3(x + self.dropout(ff_out))

        return out, attn_weights


# Decoder
class Decoder(nn.Module):
    """
    Transformer Decoder that stacks multiple DecoderLayer layers.
    """
    def __init__(self,
                 output_dim: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 dim_feedforward: int,
                 dropout: float,
                 max_len: int = 5000):
        super(Decoder, self).__init__()
        self.d_model = d_model

        # Embedding for the target tokens
        self.embed_tokens = nn.Embedding(output_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Dropout after embedding
        self.dropout = nn.Dropout(dropout)

        # Stack of DecoderLayers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        # Final linear layer to map to output vocab
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor,
                tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None):
        """
        Args:
            tgt: [batch_size, tgt_len]
            enc_out: [batch_size, src_len, d_model]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len] (future masking + pad)
            src_mask: [batch_size, 1, 1, src_len]       (source pad mask)
        Returns:
            (logits, attention) 
            logits: [batch_size, tgt_len, output_dim]
            attention: attention weights from the final DecoderLayer 
        """
        #Embed + scale
        tgt_embed = self.embed_tokens(tgt) * math.sqrt(self.d_model)

        #Add position encodings
        tgt_embed = self.pos_encoding(tgt_embed)
        tgt_embed = self.dropout(tgt_embed)

        #Pass through each layer
        out = tgt_embed
        attn_weights = None
        for layer in self.layers:
            out, attn_weights = layer(out, enc_out, tgt_mask, src_mask)

        #Project hidden states to vocab logits
        logits = self.fc_out(out)

        return logits, attn_weights

#UP NEXT (04/17/25): -> TRANSFORMER MODEL