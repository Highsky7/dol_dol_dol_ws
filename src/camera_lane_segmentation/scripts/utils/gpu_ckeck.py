import torch
# GPU 이름 체크(cuda:0에 연결된 그래픽 카드 기준)
print(torch.cuda.get_device_name(device = 0)) # 'NVIDIA TITAN X (Pascal)'

# 사용 가능 GPU 개수 체크
print(torch.cuda.device_count()) # 3