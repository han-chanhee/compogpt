import os
import numpy as np

# Read data from file
with open("combined_formatted_data2.txt", "r") as f:
    data = f.readlines()

# Extract numeric tokens and remove duplicates
numeric_tokens = set()
for line in data:
    tokens = line.split()
    for token in tokens:
        # Check if the token is a numeric value
        try:
            numeric_value = int(token)
            numeric_tokens.add(str(numeric_value))
        except ValueError:
            pass

# Convert to a list and sort for consistency
flat_data = sorted(list(numeric_tokens))

# Split into training and validation sets
n = len(flat_data)
train_data = flat_data[:int(n * 0.9)]
val_data = flat_data[int(n * 0.9):]

# Ensure an even number of elements in val_data
if len(val_data) % 2 != 0:
    val_data.pop()

# Print information about the tokens
print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")

# Export to bin files with dtype=np.uint16
train_ids = np.array(
    train_data, dtype=np.uint16
)  # Use dtype=np.uint16 to store unsigned 16-bit integers
val_ids = np.array(val_data, dtype=np.uint16)
train_ids.tofile(
    os.path.join(os.path.dirname(__file__), "train.bin"), sep="\n", format="%s"
)
val_ids.tofile(
    os.path.join(os.path.dirname(__file__), "val.bin"), sep="\n", format="%s"
)

# Print token counts
print(f"train.bin has {len(train_data):,} tokens")
print(f"val.bin has {len(val_data):,} tokens")
