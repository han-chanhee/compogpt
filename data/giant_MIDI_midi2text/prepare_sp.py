import os
import sentencepiece as spm
import numpy as np
import torch

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SentencePiece 모델 훈련 시 GPU 사용 옵션 추가
num_cpus = os.cpu_count()
spm.SentencePieceTrainer.train(input='giant_midi2.txt', model_prefix='spm_model', vocab_size=538785, num_threads=num_cpus)
sp = spm.SentencePieceProcessor()
sp.load('spm_model.model')

# Load data
with open("giant_midi2.txt", "r") as f:
    data = f.read()

# Tokenize using SentencePiece
train_tokens = sp.encode_as_pieces(data[: int(len(data) * 0.9)])
val_tokens = sp.encode_as_pieces(data[int(len(data) * 0.9) :])



# Convert tokens to IDs
train_ids = sp.encode_as_ids(data[: int(len(data) * 0.9)])
val_ids = sp.encode_as_ids(data[int(len(data) * 0.9) :])

# Move data to GPU
train_ids = torch.tensor(train_ids, dtype=torch.long).to(device)
val_ids = torch.tensor(val_ids, dtype=torch.long).to(device)

# Export to binary files directly from GPU
train_ids.cpu().numpy().astype(np.uint16).tofile("train.bin")
val_ids.cpu().numpy().astype(np.uint16).tofile("val.bin")
