import pickle

# 특수 토큰 정보
special_tokens = ["<sos>", "<eos>"]
for i in range(1, 3305):
    special_tokens.append(f"[i]")

# 특수 토큰에 대한 인덱스 매핑 생성
special_token_to_index = {token: index for index, token in enumerate(special_tokens)}
index_to_special_token = {index: token for token, index in special_token_to_index.items()}

# 메타데이터 정보 저장
meta_info = {
    "special_token_to_index": special_token_to_index,
    "index_to_special_token": index_to_special_token
}

# meta.pkl 파일에 저장
with open("meta.pkl", "wb") as f:
    pickle.dump(meta_info, f)
