import torch
import torch.nn as nn
import math


##################################
# INITIAL SETUP PROGRESS - Positional Encoding + Multi-Head Self-Attention + (Attention + Feed Forward pending)
##################################

# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    """

    The idea is to add position-dependent signals to word embeddings
    so that the model can incorporate sequence order.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.W = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.W)

    def forward(self, x):
        B, N, D = x.shape
        return x + self.W[:N, :].unsqueeze(0)



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


# Full Transformer 
class TransformerModel(nn.Module):
    """
    Complete Transformer: an Encoder and a Decoder, with the ability to
    generate source/target masks based on pad tokens or future tokens.
    """
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 d_model: int = 512, 
                 n_layers: int = 6, 
                 n_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 src_pad_idx: int = 1, 
                 tgt_pad_idx: int = 1, 
                 max_len: int = 5000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(TransformerModel, self).__init__()
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads,
                               dim_feedforward, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads,
                               dim_feedforward, dropout, max_len)

        self.to(device)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Creates a binary mask for the source sequence to ignore PAD tokens.

        Args:
            src: [batch_size, src_len]
        Returns:
            src_mask: [batch_size, 1, 1, src_len]
        """

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Creates a mask for the target sequence to:
            1) ignore PAD tokens,
            2) apply subsequent/future masking so we don't look ahead.
        
        Args:
            tgt: [batch_size, tgt_len]
        Returns:
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        """
        B, tgt_len = tgt.shape
        # Pad mask
        pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)  

        # Subsequent mask (no looking ahead)
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)  

        tgt_mask = pad_mask & subsequent_mask
        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
        Returns:
            (logits, attention)
            logits: [batch_size, tgt_len, tgt_vocab_size]
            attention: attention from the final decoder layer
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encoder(src, src_mask)
        logits, attention = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return logits, attention
