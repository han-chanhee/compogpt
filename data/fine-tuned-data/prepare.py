import os
import tiktoken
import numpy as np

# 여러 개의 스페셜 토큰 추가
special_tokens = ["<sos>", "<eos>"]
for i in range(1,3305):
    special_tokens+=f"[i]"
enc = tiktoken.get_encoding("gpt2")
special_token_ids = [enc.encode_ordinary(token)[0] for token in special_tokens]

with open("giant_midi2.txt", "r") as f:
    data = f.read()
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

# 여러 개의 스페셜 토큰 추가
train_ids.extend(special_token_ids)
val_ids.extend(special_token_ids)

print(len(set(train_ids)))
print(set(train_ids))
print(len(set(val_ids)))
print(set(val_ids))
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# train.bin has 301,970 tokens
# val.bin has 36,063 tokens
