"""
convert_onnx_to_pth.py
----------------------
과제 #3의 mnist_fc.onnx 가중치를 α,β-CROWN이 직접 로드할 수 있는 PyTorch .pth로 옮긴다.

배경: 이 ONNX는 가중치를 external-data로 저장하고 맨 앞에 Reshape 노드가 있어
onnx2pytorch 변환이 멈춘다. 따라서 ONNX initializer에서 가중치를 직접 읽어
동일 구조의 PyTorch 모듈에 적재한 뒤 state_dict로 저장한다.

주의: custom/mnist_fc_model.py의 mnist_fc()는 'net.' 접두사 없는 순수 nn.Sequential
(키 0./2./4.)을 반환하므로, 저장 시에도 같은 키 형식이어야 한다. SmallFC는 net으로
감싸므로 'net.' 접두사를 제거하고 저장한다.

저장 위치:
  - mnist_fc.pth  (레포 루트, git 추적용)
  검증기가 로드하는 위치로의 복사는 README의 셋업 단계를 참고.
"""
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper

ONNX_PATH = "mnist_fc.onnx"
ROOT_PTH = "mnist_fc.pth"


class SmallFC(nn.Module):
    """과제 #3 학습 코드와 동일 구조 (net으로 감싼 Sequential)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 32), nn.ReLU(),
            nn.Linear(32, 16),  nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# --- ONNX initializer에서 가중치 읽어 PyTorch 모듈에 적재 ---
m = onnx.load(ONNX_PATH)
inits = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}

model = SmallFC()
sd = model.state_dict()
# 이름이 그대로 일치(net.0.weight 등)하므로 직접 매핑. val_3 등 비-파라미터는 제외.
new_sd = {k: torch.from_numpy(inits[k].copy()) for k in sd.keys()}
model.load_state_dict(new_sd)
model.eval()

# --- 검증: ONNX 추론과 PyTorch 추론이 일치하는지 확인 ---
import onnxruntime as ort
x = np.random.rand(1, 784).astype(np.float32)
ort_out = ort.InferenceSession(ONNX_PATH).run(None, {"input": x})[0]
pt_out = model(torch.from_numpy(x)).detach().numpy()
print("max abs diff (onnx vs pytorch):", np.abs(ort_out - pt_out).max())

# --- 'net.' 접두사 제거: net.0.weight -> 0.weight ---
# custom/mnist_fc_model.py의 mnist_fc()가 반환하는 순수 Sequential과 키를 맞춘다.
stripped = {k.replace("net.", ""): v for k, v in model.state_dict().items()}
print("saved keys:", list(stripped.keys()))

# --- 레포 루트에 저장 ---
torch.save(stripped, ROOT_PTH)
print(f"saved {ROOT_PTH}")