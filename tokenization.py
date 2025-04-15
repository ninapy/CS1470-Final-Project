import torch
import spacy #nlp for tokenization library
import io #just for file operations like .open
import os
from collections import Counter #for token frequency

try:
    spacy_en = spacy.load("en_core_web_sm") #eng model
    spacy_de = spacy.load("de_core_news_sm") #deutsch model
except OSError: #download them if they are not alr
    print("Downloading spaCy models")
    import os
    os.system("python -m spacy download en_core_web_sm")
    os.system("python -m spacy download de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")
    spacy_de = spacy.load("de_core_news_sm")

#Define tokenizers
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

SRC_LANGUAGE = 'de' #for Deutsch
TGT_LANGUAGE = 'en'
SOS_TOKEN = '<sos>'  #start of sentence (decoder input)
EOS_TOKEN = '<eos>'  #end of sentence
PAD_TOKEN = '<pad>'  #padding
UNK_TOKEN = '<unk>'  #unknown token if its an out-of-vocabulary words

#Custom vocabulary class to replace torchtext's vocab functionality (bc its deprecated)
class Vocabulary:
    """
    Vocabulary class that maps tokens to indices and vice versa.
    Keeps track of token frequencies and handles special tokens.
    """
    def __init__(self, specials=None, default_index=None):
        """
        Initialize vocabulary with optional special tokens.
        
        Args:
            specials: List of special tokens to add (e.g., <unk>, <pad>)
            default_index: Default index to return for unknown tokens
        """
        self.token_to_idx = {}  #maps tokens to their indices
        self.idx_to_token = {}  #maps indices back to tokens
        self.frequencies = {}   #tracks frequency of each token
        
        #add special tokens if provided
        if specials:
            for token in specials:
                self.add_token(token, special=True)
        
        #set default index for unknown tokens
        self.default_index = default_index
    
    def add_token(self, token, special=False, freq=None):
        """
        Add a token to the vocabulary or update its frequency.
        
        Args:
            token: The token to add
            special: Whether this is a special token (e.g., <unk>)
            freq: Optional frequency to assign to this token
        """
        if token not in self.token_to_idx:
            #if token doesn't exist, add it with the next available index
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            #set frequency based on parameters
            if freq is not None:
                self.frequencies[token] = freq
            elif special:
                self.frequencies[token] = float('inf')#special tokens have inf freq
            else:
                self.frequencies[token] = 1
        elif freq is not None:
            self.frequencies[token] = freq
        elif not special:
            self.frequencies[token] += 1
    
    def __getitem__(self, token):
        """
        Get the index for a token. If token is not in vocabulary,
        return the default index if set, otherwise raise KeyError.
        
        Args:
            token: The token to look up
            
        Returns:
            The index of the token
        """
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        elif self.default_index is not None:
            return self.default_index
        raise KeyError(f"Token '{token}' not found in vocabulary")
    
    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.token_to_idx)
    
    def set_default_index(self, idx):
        """Set the default index to return for unknown tokens."""
        self.default_index = idx
    
    def get_token(self, idx):
        """Get the token for an index, returning UNK_TOKEN if not found."""
        return self.idx_to_token.get(idx, UNK_TOKEN)

#read tokens from file and count frequencies
def count_tokens(data_path, tokenizer):
    """
    Count token frequencies in a file.
    
    Args:
        data_path: Path to text file
        tokenizer: Function to tokenize each line
        
    Returns:
        Counter object with token frequencies
    """
    counter = Counter()
    with io.open(data_path, encoding='utf8') as f:
        for line in f:
            #tokenize each line and update counter
            tokens = tokenizer(line.rstrip())  #rstrip to remove trailing whitespace
            counter.update(tokens)
    return counter

# Build vocabulary from token counts
def build_vocabulary(train_filepaths, tokenizers, min_freq=2):
    """
    Build vocabularies for each language from training files.
    
    Args:
        train_filepaths: Dictionary mapping language codes to file paths
        tokenizers: Dictionary mapping language codes to tokenizer functions
        min_freq: Minimum frequency for a token to be included
        
    Returns:
        Dictionary mapping language codes to Vocabulary objects
    """
    vocab_transforms = {}
    
    for ln, tokenizer in tokenizers.items():
        train_filepath = train_filepaths[ln]
        print(f"Building vocabulary for {ln} using {train_filepath}")
        
        #initialize vocabulary with special tokens
        specials = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        vocab = Vocabulary(specials=specials)
        
        #count token frequencies
        counter = count_tokens(train_filepath, tokenizer)
        
        #add tokens that meet minimum frequency
        for token, count in counter.items():
            if count >= min_freq:  #only include tokens that appear at least min_freq times
                vocab.add_token(token, freq=count)
        
        #set default index to UNK token
        vocab.set_default_index(vocab[UNK_TOKEN])
        
        vocab_transforms[ln] = vocab
        print(f"Vocabulary size for {ln}: {len(vocab)}")
        
    return vocab_transforms

train_filepaths = {
    'en': 'data/training/train.en',
    'de': 'data/training/train.de'
}

#define tokenizers for each language
token_transforms = {
    'en': tokenize_en,  #eng tokenizer
    'de': tokenize_de   #de tokenizer
}

#create sequential transform function
def sequential_transforms(*transforms):
    """
    Create a function that applies a sequence of transforms.
    
    Args:
        *transforms: Variable number of transform functions
        
    Returns:
        Function that applies all transforms in sequence
    """
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

#function to create tensors with SOS/EOS tokens
def tensor_transform(vocab, token_ids):
    """
    Convert token indices to a tensor and add SOS/EOS tokens.
    
    Args:
        vocab: The vocabulary to use for SOS/EOS token indices
        token_ids: List of token indices
        
    Returns:
        Tensor with SOS at beginning and EOS at end
    """
    return torch.cat((
        torch.tensor([vocab[SOS_TOKEN]]),       #start token
        torch.tensor(token_ids),                #actual content
        torch.tensor([vocab[EOS_TOKEN]])        #end token
    ))

#build vocabularies
vocab_transforms = build_vocabulary(train_filepaths, token_transforms, min_freq=2)

#create text transformers for each language
text_transforms = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transforms[ln] = sequential_transforms(
        token_transforms[ln],                   #first tokenize the text
        #then convert tokens to indices using the vocabulary
        lambda tokens: [vocab_transforms[ln][token] for token in tokens],
        #finally add SOS/EOS and create tensor
        lambda token_ids: tensor_transform(vocab_transforms[ln], token_ids)
    )
 # Example usage with full pipeline
# Example usage - Test the transformation pipeline
de_text = "Zwei Männer unterhalten sich, während sie in einer Küche kochen."
en_text = "Two men are talking while they are cooking in a kitchen."
# Apply transformation pipelines to convert text to token indices
de_tokens = text_transforms['de'](de_text)
en_tokens = text_transforms['en'](en_text)

print(f"German tokens: {de_tokens}")
print(f"English tokens: {en_tokens}")
#example:
# <unk>: 0
# <pad>: 1
# <sos>: 2
# <eos>: 3
# 'the': 4
# 'cat': 5
# '.': 6
# 'dog': 7
#Then the sentence "The dog saw the cat." would become:
#[2, 4, 7, 0, 4, 5, 6, 3]

#TODO:
# Create a dataset class that loads all the data
# Implement batching with padding for variable-length sequences