import os
import torch
from contextlib import nullcontext
import tiktoken
from model import GPTConfig, GPT

# 초기 설정
init_from = "resume"  # 'resume' (from an out_dir) 또는 GPT-2 변형 모델 (예: 'gpt2-xl')
out_dir = "out"
start = "<sos>"  # 또는 "" 또는 기타 설정할 시작 텍스트
num_samples = 100  # 생성할 샘플 수
max_new_tokens = 5000  # 각 샘플당 생성할 최대 토큰 수
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
compile = False

# 모델 초기화
torch.manual_seed(seed)
device_type = "cuda" if "cuda" in device else "cpu"
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type)

gptconf = GPTConfig()  # GPTConfig에 필요한 추가적인 설정이 있다면 여기에 추가하세요
model = GPT(gptconf)
model_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval().to(device)
if compile:
    model = torch.compile(model)

# 문장 생성 함수
def generate_text():
    generated_samples_count = 0
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                generated_text = ""
                for _ in range(max_lines):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    lines = decode(y[0].tolist()).strip().split('\n')
                    for line in lines:
                        # 각 줄에 "[" 문자가 나오면 숫자와 "]" 문자를 붙이기
                        if "[" in line:
                            line = line.replace("[", f"[{generated_samples_count + 1}]")
                            generated_samples_count += 1
                        generated_text += f"{line}\n"

                        # 결과물을 파일로 저장
                        output_path = os.path.join(output_dir, f"sample_{generated_samples_count}.txt")
                        with open(output_path, "w") as file:
                            file.write(line + "\n")

                        # 필요한 만큼 결과물을 출력하고 200줄을 넘으면 중단
                        print(generated_text)
                        print("---------------")
                        if generated_samples_count >= max_lines:
                            return

# 모델 관련 설정
x = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]
meta_path = os.path.join("data", "meta.pkl")
load_meta = os.path.exists(meta_path)
if load_meta:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<eos>"})
    decode = lambda l: enc.decode(l)

# 결과물 저장 디렉토리 생성
output_dir = "generated_samples"
os.makedirs(output_dir, exist_ok=True)

# 텍스트 생성 (최대 200줄까지 생성)
max_lines = 200
generate_text()
