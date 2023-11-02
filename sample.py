import os
import pickle
import torch
import tiktoken
from model import GPTConfig, GPT
from contextlib import nullcontext

# -----------------------------------------------------------------------------
init_from = "resume"
out_dir = "out-midi"
start = "FILE:prompt.txt"
num_samples = 10
max_total_tokens = 200
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# 모델 초기화
if init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# 파일에서 시작 텍스트 읽어오기
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()

# 인코딩 및 디코딩 함수 정의
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: "".join([enc.decode([i]) for i in l])

# 생성된 샘플 저장 폴더 생성
output_dir = out_dir
os.makedirs(output_dir, exist_ok=True)

# 샘플 생성 및 저장
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            generated_tokens = []
            while len(generated_tokens) < max_total_tokens:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                generated_tokens.extend(y[0].tolist())
                if "[" in enc.decode([y[0][-1]]):
                    break

            generated_text = decode(generated_tokens)
            output_path = os.path.join(output_dir, f"sample_{k+1}.txt")
            with open(output_path, "w") as file:
                file.write(generated_text)
