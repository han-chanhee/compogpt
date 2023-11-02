"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = (
    "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
)
out_dir = "out"  # ignored if init_from is not 'resume'
start = "<SOS>"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = (
    0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
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

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    init_from == "resume"
    and "config" in checkpoint
    and "dataset" in checkpoint["config"]
):  # older checkpoints might not have these...
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<EOS>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# decode 함수를 수정하여 "[" 뒤의 숫자와 "]"를 붙이는 부분입니다.
def decode(tokens):
    text = ""
    open_bracket = False
    number_to_append = ""
    for token in tokens:
        token_str = itos[token]
        if token_str == "[":
            open_bracket = True
            number_to_append = "1"  # 숫자 초기값 설정
        elif token_str.isdigit() and open_bracket:
            number_to_append = str(int(number_to_append) + 1)  # 숫자를 1씩 증가시킴
        elif token_str == "]" and open_bracket:
            text += number_to_append + "]"
            open_bracket = False
            number_to_append = ""
        else:
            text += token_str
    return text

# 모델 초기화 및 생성된 샘플 저장 부분입니다.
num_samples = 10

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            generated_tokens = []
            while len(generated_tokens) < max_total_tokens:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                generated_tokens.extend(y[0].tolist())
                if "[" in itos[y[0][-1]]:
                    generated_tokens[-1] += encode("[" + number_to_append + "]")
                    break
                else:
                    number_to_append = ""  # 숫자와 "]"를 붙이기 위해 사용된 변수 초기화
                    for token in reversed(y[0].tolist()):
                        if token == itos.index("["):  # "["가 나오면 루프 탈출
                            break
                        elif token.isdigit():  # 숫자일 경우 number_to_append에 추가
                            number_to_append = itos[token] + number_to_append
                    generated_tokens[-1] += encode("[" + number_to_append + "]")

            generated_text = decode(generated_tokens)
            
            # 각 샘플을 파일로 저장
            output_dir = "generated_samples"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"sample_{k+1}.txt")
            with open(output_path, "w") as file:
                file.write(generated_text)
