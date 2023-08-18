import os
import sentencepiece as spm
import numpy as np

# Load or train SentencePiece model
spm.SentencePieceTrainer.train(input='giant_midi.txt', model_prefix='spm_model', vocab_size=1000)
sp = spm.SentencePieceProcessor()
sp.load('spm_model.model')

# Load data
with open("giant_midi.txt", "r") as f:
    data = f.read()

# Tokenize using SentencePiece
train_tokens = sp.encode_as_pieces(data[: int(len(data) * 0.9)])
val_tokens = sp.encode_as_pieces(data[int(len(data) * 0.9) :])

# Count unique tokens
unique_train_tokens = len(set(train_tokens))
unique_val_tokens = len(set(val_tokens))
print(len(unique_train_tokens))
print("Unique Train Tokens:", unique_train_tokens)
print(len(unique_val_tokens))
print("Unique Val Tokens:", unique_val_tokens)

# Convert tokens to IDs
train_ids = sp.encode_as_ids(data[: int(len(data) * 0.9)])
val_ids = sp.encode_as_ids(data[int(len(data) * 0.9) :])

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
