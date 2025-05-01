import torch
import sounddevice as sd
import soundfile as sf
import spacy
import numpy as np
import os
from gtts import gTTS
from IPython.display import Audio, display

import json
import pickle

# ---- Device Config ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Special Tokens ----
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# ---- Vocabulary Class ----
class Vocabulary: #build vocabulary class to map tokens to unique indices
    def __init__(self, specials=None):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.frequencies = {}

        #add special tokens like <unk>, <pad>, <sos>, <eos>
        if specials:
            for token in specials:
                self.add_token(token, special=True)

    def add_token(self, token, special=False):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            self.frequencies[token] = float('inf') if special else 1
        elif not special:
            self.frequencies[token] += 1

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx[UNK_TOKEN])

    def __len__(self):
        return len(self.token_to_idx)

    def get_token(self, idx):
        return self.idx_to_token.get(idx, UNK_TOKEN)

    def get_stoi(self):
        return self.token_to_idx
    
import torch.serialization
torch.serialization.add_safe_globals({'Vocabulary': Vocabulary})

# ---- Tokenizers ----
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_de(text):
    return [token.text.lower() for token in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [token.text.lower() for token in spacy_en.tokenizer(text)]

# ---- Model Components ----
import torch.nn as nn
import math

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len).expand((1, -1)))

    def forward(self, x): #add positional embeddings to token embeddings
        """
        x: (batch_size, seq_len, d_model)
        """
        position_ids = self.position_ids[:, :x.size(1)].to(x.device)
        return x + self.pe(position_ids)

#multi-head self-attention layer
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention sublayer.

    Given queries, keys, and values (Q, K, V), it produces an output
    by performing scaled dot-product attention in multiple heads, then
    concatenates the results.
    """
    # allows the model to focus on different parts of the sequence

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        #define linear layers for query, key, and value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        #output projection
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        #apply linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        #split into multiple heads
        Q = Q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        #compute scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        #apply attention to values
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, -1, self.d_model)

        #final linear projection
        x = self.fc_out(x)
        return x, attention

#position-wise feed-forward network
class PositionwiseFeedForward(nn.Module):
    """
    Implements the two-layer feed-forward network used in the Transformer:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x): #apply two linear transformations with relu activation
        # Project up -> apply ReLU -> dropout -> project down
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

#single encoder layer with self-attention and feed-forward
class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer, made up of:
      - Multi-Head Self-Attention (with residual & layer norm)
      - Position-wise Feed Forward (with residual & layer norm)
    """
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        #apply self-attention sublayer
        attn_output, _ = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))

        #apply feed-forward sublayer
        ff_output = self.ff(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src

#encoder consisting of multiple encoder layers
class Encoder(nn.Module):
    """
    Transformer Encoder that stacks multiple EncoderLayer layers.
    """
    def __init__(self, input_dim, d_model, n_layers, n_heads, dim_feedforward, dropout, max_len=5000):
        super().__init__()
        self.d_model = d_model

        #token embedding
        self.embed_tokens = nn.Embedding(input_dim, d_model)

        #positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len)

        #dropout after embedding
        self.dropout = nn.Dropout(dropout)

        #stack of EncoderLayers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src, src_mask=None):
        #embed tokens + scale
        src_embed = self.embed_tokens(src) * math.sqrt(self.d_model)
        #add positional encoding
        src_embed = self.pos_encoding(src_embed)
        #apply dropout
        src_embed = self.dropout(src_embed)

        #pass through each layer
        out = src_embed
        for layer in self.layers:
            out = layer(out, src_mask)

        return out

#single decoder layer with masked self-attention and encoder-decoder attention
class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer, made up of:
      - Masked Multi-Head Self-Attention (with residual & layer norm)
      - Multi-Head Attention over Encoder output (with residual & layer norm)
      - Position-wise Feed Forward (with residual & layer norm)
    """
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        # Self-attention for the target tokens
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # Cross-attention: query is the decoder, key/value is the encoder output
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        # Masked self-attention
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        x = self.norm1(tgt + self.dropout(self_attn_out))

        # Encoder-Decoder cross-attention
        cross_attn_out, attn_weights = self.cross_attn(x, enc_out, enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Position-wise feed forward
        ff_out = self.ff(x)
        out = self.norm3(x + self.dropout(ff_out))

        return out, attn_weights

#decoder consisting of multiple decoder layers
class Decoder(nn.Module):
    """
    Transformer Decoder that stacks multiple DecoderLayer layers.
    """
    def __init__(self, output_dim, d_model, n_layers, n_heads, dim_feedforward, dropout, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # Embedding for the target tokens
        self.embed_tokens = nn.Embedding(output_dim, d_model)

        # Positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len)

        # Dropout after embedding
        self.dropout = nn.Dropout(dropout)

        # Stack of DecoderLayers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        # Final linear layer to map to output vocab
        self.fc_out = nn.Linear(d_model, output_dim, bias=False)
        self.embed_tokens = nn.Embedding(output_dim, d_model)

        # Share weights
        self.fc_out.weight = self.embed_tokens.weight

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        # Embed + scale
        tgt_embed = self.embed_tokens(tgt) * math.sqrt(self.d_model)

        # Add position encodings
        tgt_embed = self.pos_encoding(tgt_embed)
        tgt_embed = self.dropout(tgt_embed)

        # Pass through each layer
        out = tgt_embed
        attn_weights = None
        for layer in self.layers:
            out, attn_weights = layer(out, enc_out, tgt_mask, src_mask)

        # Project hidden states to vocab logits
        logits = self.fc_out(out)

        return logits, attn_weights

#complete transformer model combining encoder and decoder
class Transformer(nn.Module):
    """
    Complete Transformer: an Encoder and a Decoder, with the ability to
    generate source/target masks based on pad tokens or future tokens.
    """
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 n_layers=6,
                 n_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 src_pad_idx=1,
                 tgt_pad_idx=1,
                 max_len=5000):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads,
                              dim_feedforward, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads,
                              dim_feedforward, dropout, max_len)

        #initialize parameters with xavier uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src): #create mask for source to ignore pad tokens
        """
        Creates a binary mask for the source sequence to ignore PAD tokens.
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        Creates a mask for the target sequence to:
            1) ignore PAD tokens,
            2) apply subsequent/future masking so we don't look ahead.
        """
        B, tgt_len = tgt.shape
        # Pad mask
        pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        # Subsequent mask (no looking ahead)
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)

        tgt_mask = pad_mask & subsequent_mask
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encoder(src, src_mask)
        logits, attention = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return logits, attention

# ---- Load Model ----
def load_model(filename, device):
    #checkpoint = torch.load(filename, map_location=device)
    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    # Get vocabularies
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']

    # Get hyperparameters
    hyperparams = checkpoint.get('model_hyperparams', {
        'd_model': 256,
        'n_layers': 6,
        'n_heads': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1
    })

    # Initialize model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=hyperparams['d_model'],
        n_layers=hyperparams['n_layers'],
        n_heads=hyperparams['n_heads'],
        dim_feedforward=hyperparams['dim_feedforward'],
        dropout=hyperparams['dropout'],
        src_pad_idx=src_vocab[PAD_TOKEN],
        tgt_pad_idx=tgt_vocab[PAD_TOKEN]
    ).to(device)

    # Load model parameters
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, src_vocab, tgt_vocab

def load_model_separately(weight_path, vocab_prefix, hyperparam_path, device):
    # Load vocabularies
    with open(f"{vocab_prefix}_src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open(f"{vocab_prefix}_tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)

    # Load hyperparameters
    with open(hyperparam_path, "r") as f:
        params = json.load(f)

    # Rebuild model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=params['d_model'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        dim_feedforward=params['dim_feedforward'],
        dropout=params['dropout'],
        src_pad_idx=src_vocab[PAD_TOKEN],
        tgt_pad_idx=tgt_vocab[PAD_TOKEN]
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    return model, src_vocab, tgt_vocab

# ---- Translate Sentence ----
def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_len=50, return_attention=False):
    model.eval()

    if isinstance(sentence, str):
        tokens = [SOS_TOKEN] + tokenize_de(sentence) + [EOS_TOKEN]
    else:
        tokens = [SOS_TOKEN] + sentence + [EOS_TOKEN]

    src_indices = [src_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    tgt_indices = [tgt_vocab[SOS_TOKEN]]
    outputs = []
    attentions = []

    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)

        with torch.no_grad():
            output, attention = model.decoder(tgt_tensor, enc_src, tgt_mask, src_mask)

        output = output[:, -1, :]  # (batch_size=1, vocab_size)
        pred_token = output.argmax(1).item()

        outputs.append(pred_token)
        attentions.append(attention)

        if pred_token == tgt_vocab[EOS_TOKEN]:
            break

        tgt_indices.append(pred_token)

    pred_tokens = [tgt_vocab.get_token(idx) for idx in outputs]

    if return_attention:
        # return final step attention
        return pred_tokens, attentions[-1]
    else:
        return pred_tokens, None
    
def translate_custom_sentence(sentence):
    #loaded_model, loaded_src_vocab, loaded_tgt_vocab = load_model('model/transformer_full_model.pt', DEVICE)
    loaded_model, loaded_src_vocab, loaded_tgt_vocab = load_model_separately(
        weight_path="model/transformer-model-best_weights.pt",
        vocab_prefix="model/transformer-model-best",
        hyperparam_path="model/transformer-model-best_hyperparams.json",
        device=DEVICE
    )

    # Tokenize the sentence
    tokens = tokenize_de(sentence)

    # Print original tokens
    print(f"Original: {sentence}")
    print(f"Tokenized: {tokens}")

    # Translate
    translation, attention = translate_sentence(loaded_model, tokens, loaded_src_vocab, loaded_tgt_vocab, DEVICE, return_attention=True)

    # Remove EOS token if present
    if EOS_TOKEN in translation:
        translation = translation[:translation.index(EOS_TOKEN)]

    # Print translation
    print(f"Translation: {' '.join(translation)}")

    return translation, attention

# ---- Audio Recording ----
def record_audio(duration=5, sample_rate=16000):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    filename = "temp_recording.wav"
    sf.write(filename, recording, sample_rate)
    return filename

# ---- Transcribe Audio ----
from faster_whisper import WhisperModel

def transcribe_audio(audio_file, language="de"):
    print("Transcribing audio with faster-whisper...")
    model = WhisperModel("base", device="cpu")
    segments, _ = model.transcribe(audio_file, language=language)
    return "".join(segment.text for segment in segments).strip()

# ---- Text-to-Speech ----
def speak_text_online(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    temp_file = "temp_output.mp3"
    tts.save(temp_file)
    os.system(f'start {temp_file}')

# ---- Full Pipeline ----
def speech_to_speech_translation(return_text=False):
    print("Say a German sentence...")
    audio_file = record_audio(duration=5)
    german_text = transcribe_audio(audio_file)
    print(f"German: {german_text}")

    translation_tokens, _ = translate_custom_sentence(german_text)
    translated_text = ' '.join(translation_tokens)
    print(f"English: {translated_text}")

    # Generate audio
    os.makedirs("static", exist_ok=True)
    audio_path = "static/translated_audio.mp3"
    tts = gTTS(text=translated_text, lang='en')
    tts.save(audio_path)

    if return_text:
        return german_text, translated_text, audio_path
    else:
        os.system(f'start {audio_path}')

if __name__ == "__main__":
    translate_custom_sentence("Ein Mann spielt Gitarre.")
    translate_custom_sentence("Zwei Kinder spielen im Park.")
    translate_custom_sentence("Die Frau liest ein Buch im Café.")
    translate_custom_sentence("Ein Kind rennt im Park.")
    translate_custom_sentence("Die Frau liest ein Buch.")
    translate_custom_sentence("Zwei Hunde spielen im Garten.")
    translate_custom_sentence("Eine Katze schläft auf dem Sofa.")

    #speech_to_speech_translation()
