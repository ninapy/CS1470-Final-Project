import torch
# from torch.utils.data import Dataset #base for custom datasets
# from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator #count token freq
import spacy #nlp for tokenization library
import io #just for file operations like .open

try:
    spacy_en = spacy.load("en_core_web_sm")
    spacy_de = spacy.load("de_core_news_sm")
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
SOS_TOKEN = '<sos>'  # Start of sentence (decoder input)
EOS_TOKEN = '<eos>'  # End of sentence
PAD_TOKEN = '<pad>'  # Padding
UNK_TOKEN = '<unk>'  # Unknown token if its an out-of-vocabulary words

#Build vocabulary
def yield_tokens(data_path, tokenizer, index):
    with io.open(data_path, encoding='utf8') as f:
        for line in f:
            #yield for mem efficiency bc it processes one line at the time
            yield tokenizer(line.rstrip()) #rstrip to strip white space

def build_vocabulary(train_filepaths, tokenizers):
    vocab_transforms = {}
    
    for ln, tokenizer in tokenizers.items():
        train_filepath = train_filepaths[ln]
        vocab_transforms[ln] = build_vocab_from_iterator( #count token freq
            yield_tokens(train_filepath, tokenizer, ln), #makes the tokens in the language(en or de) and specific train data
            min_freq=2,#only words that appear at least twice
            specials=[UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN],#adds the special tokens in the beginning
            special_first=True
        )
        #Set the default index to UNK_TOKEN idx
        vocab_transforms[ln].set_default_index(vocab_transforms[ln][UNK_TOKEN])
        #1st iteration in english, 2nd in Deutsch and then returns the vocab transformations
    return vocab_transforms

# Define file paths
train_filepaths = {
    'en': 'data/training/train.en',
    'de': 'data/training/train.de'
}

# Define tokenizers for each language
token_transforms = {
    'en': tokenize_en,
    'de': tokenize_de
}

# Build vocabularies
#creates dictionaries (one per language) that map tokens to indices like "Zwei" → 12
vocab_transforms = build_vocabulary(train_filepaths, token_transforms)

# Create tensor transforms
#returns a new function that applies each transform in sequence
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((
        torch.tensor([vocab_transforms[SRC_LANGUAGE][SOS_TOKEN]]),
        torch.tensor(token_ids),
        torch.tensor([vocab_transforms[SRC_LANGUAGE][EOS_TOKEN]])
    ))

# Text transforms for source and target languages
text_transforms = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transforms[ln] = sequential_transforms(
        token_transforms[ln],
        # Convert tokens to indices using the vocabulary
        lambda tokens: [vocab_transforms[ln][token] for token in tokens],
        # Add SOS/EOS and create tensor
        tensor_transform
    )

# Example usage
de_text = "Zwei Männer unterhalten sich, während sie in einer Küche kochen."
en_text = "Two men are talking while they are cooking in a kitchen."

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