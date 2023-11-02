import os
import pickle
import torch
import tiktoken
from model import GPTConfig, GPT
from contextlib import nullcontext

# 초기 설정
init_from = "resume"  # 'resume' 또는 GPT-2 모델 (예: 'gpt2-xl')
out_dir = "out-midi"
start = "<SOS>"
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
compile = False

# 랜덤 시드 설정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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

# 문장 생성 및 대괄호 안에 숫자 넣기
generated_lines_with_brackets = []  
count = 1  
with torch.no_grad():
    with nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype):
        for _ in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = decode(y[0].tolist())

            # 대괄호가 있는 라인 처리
            lines = generated_text.split('\n')
            for line in lines:
                if '[' in line and ']' in line:
                    line_with_bracket = line.replace('[', f'[{count}]', 1)
                    count += 1
                    generated_lines_with_brackets.append(line_with_bracket)
                else:
                    generated_lines_with_brackets.append(line)

                # 생성된 라인 출력
                print(line_with_bracket)
                print("---------------")

# 대괄호가 있는 라인 개수 세기
num_lines_with_brackets = len(generated_lines_with_brackets)
print(f"대괄호가 있는 라인 개수: {num_lines_with_brackets}")

# 대괄호가 있는 라인 저장
output_with_brackets_path = os.path.join(output_dir, "lines_with_brackets.txt")
with open(output_with_brackets_path, "w") as file:
    for line in generated_lines_with_brackets:
        file.write(line + '\n')
