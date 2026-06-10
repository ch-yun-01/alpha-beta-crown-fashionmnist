"""ONNX weight를 SmallFC PyTorch 모델로 옮겨 .pth로 저장."""
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper

class SmallFC(nn.Module):
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

m = onnx.load("alpha-beta-CROWN/complete_verifier/models/mnist_fc.onnx")
inits = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}

model = SmallFC()
sd = model.state_dict()
# 이름이 그대로 일치하므로 직접 매핑 (val_3 등 비-파라미터는 제외)
new_sd = {k: torch.from_numpy(inits[k].copy()) for k in sd.keys()}
model.load_state_dict(new_sd)
model.eval()

# 검증: ONNX 추론과 PyTorch 추론이 일치하는지 확인
import numpy as np
import onnxruntime as ort
x = np.random.rand(1, 784).astype(np.float32)
ort_out = ort.InferenceSession(
    "alpha-beta-CROWN/complete_verifier/models/mnist_fc.onnx"
).run(None, {"input": x})[0]
pt_out = model(torch.from_numpy(x)).detach().numpy()
print("max abs diff (onnx vs pytorch):", np.abs(ort_out - pt_out).max())

torch.save(model.state_dict(), "alpha-beta-CROWN/complete_verifier/models/mnist_fc.pth")
print("saved mnist_fc.pth")
