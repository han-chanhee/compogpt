import os
import numpy as np

# Read data from file
with open("mididata.txt", "r") as f:
    data = f.readlines()

# Flatten the list of lines to make each line a separate "token"
flat_data = [token.strip() for line in data for token in line.split()]

# Split into training and validation sets
n = len(flat_data)
train_data = flat_data[: int(n * 0.9)]
val_data = flat_data[int(n * 0.9) :]

# Print information about the tokens
print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")

# Export to bin files
train_ids = np.array(train_data, dtype=np.object)  # Use dtype=np.object to store strings
val_ids = np.array(val_data, dtype=np.object)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"), sep="\n")
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"), sep="\n")

# Print token counts
print(f"train.bin has {len(train_data):,} tokens")
print(f"val.bin has {len(val_data):,} tokens")
