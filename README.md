# 과제 #4 — α,β-CROWN을 이용한 신경망 강건성 검증

신뢰할 수 있는 인공지능 (Reliable and Trustworthy Artificial Intelligence)

본 저장소는 [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) 검증기를
사용하여 두 개의 ReLU 신경망에 대한 국소 L∞ 강건성(local robustness)을 검증한다.
섭동 반경 ε를 변화시키며 검증 난이도가 어떻게 달라지는지 분석하고, 과제 #3에서 사용한
**동일한** MNIST 완전연결 신경망에 대해 Marabou(SMT 기반 검증기)와 결과를 비교한다.

## 검증 대상 모델

두 모델 모두 α,β-CROWN에 기본 포함되지 않은 외부 모델이므로 "external model" 요건을 충족한다.

| 모델 | 구조 | 데이터셋 | Clean Acc. | 역할 |
|------|------|----------|-----------|------|
| **FashionMNIST CNN** | Conv(1→16,4×4,s2)–ReLU–Conv(16→32,4×4,s2)–ReLU–FC(1568→100)–ReLU–FC(100→10) | FashionMNIST | 90.84% | 외부 합성곱 모델; ε 스윕 + 확장성 |
| **MNIST FC** | FC(784→32)–ReLU–FC(32→16)–ReLU–FC(16→10) | MNIST | ~95% | 과제 #3과 동일 모델 → Marabou 비교 |

MNIST FC의 가중치는 과제 #3 저장소의 `mnist_fc.onnx`를 그대로 재사용하여, 두 검증기가
완전히 동일한 신경망에 대해 실행되도록 하였다.

## 검증 속성 (Verification Property)

첫 50개 테스트 이미지 `x`(실제 레이블 `y`)에 대해, ε 반경의 L∞ 볼(유효 픽셀 범위
`[0,1]`로 클리핑) 안의 모든 `x'`가 여전히 `y`로 분류되는지 검증한다:

```
∀ x'.  max|x' − x| ≤ ε   ⇒   argmax f(x') = y
```

두 모델 모두 입력 정규화(normalization)를 사용하지 않으므로 ε는 픽셀 공간에서 직접 표현된다.
각 실행은 빠른 반례 탐색을 위한 PGD 공격, 불완전 검증을 위한 α-CROWN, 그리고 β-CROWN
분기한정(branch-and-bound, kfsb 분기)을 사용하며, 인스턴스당 120초의 timeout을 둔다.

### Epsilon 스윕

- **FashionMNIST CNN:** ε ∈ {1, 2, 4, 6, 8, 12, 16} / 255
- **MNIST FC:** ε ∈ {0.01, 0.03, 0.05, 0.1, 0.2}  (과제 #3과 동일)

## 저장소 구조

```
.
├── README.md
├── requirements.txt
├── report.pdf                  # 보고서: 배경, 결과, 분석, Marabou 비교
├── test.py                     # configs/ 전체(또는 --config 하나) 실행 후 결과 파싱
├── train_model.py              # FashionMNIST CNN 학습
├── convert_onnx_to_pth.py      # mnist_fc.onnx 가중치를 PyTorch(.pth)로 이전
├── fashion_mnist_model.py      # FashionMNIST 모델 + α,β-CROWN용 커스텀 데이터로더
├── mnist_fc_model.py           # MNIST FC 모델 정의 (nn.Sequential 반환)
├── mnist_fc.onnx               # 과제 #3의 MNIST FC 가중치 (external-data ONNX)
├── mnist_fc.onnx.data          # 위 ONNX의 외부 가중치 blob
├── make_comparison_figs.py     # 보고서용 비교 figure 생성
├── configs/
│   ├── fashion_mnist_eps_{1,2,4,6,8,12,16}_255.yaml
│   └── mnist_fc_eps_{0.01,0.03,0.05,0.1,0.2}.yaml
└── results/
    └── <config_name>/
        ├── results_summary.txt
        └── log_<config_name>.txt
```

클론한 검증기(`alpha-beta-CROWN/`)와 다운로드된 데이터셋은 git에서 제외된다(.gitignore).

## 설치 (Setup)

```bash
# 1. 검증기를 서브모듈과 함께 현재 디렉터리로 클론
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git

# 2. 검증기에 포함된 conda 환경 생성
conda env create -f alpha-beta-CROWN/complete_verifier/environment.yaml
conda activate alpha-beta-crown

# 3. 보조 스크립트에 필요한 추가 의존성 설치
pip install -r requirements.txt
```

## 실험 재현 (Reproducing)

아래 1–2단계로 모델을 준비한 뒤, 3단계의 `python test.py` 한 번으로 두 모델의 모든
config(총 12개)를 순차 실행하여 결과를 생성한다.

### 1단계: FashionMNIST CNN 준비

```bash
# 모델 학습 (fashion_mnist_cnn.pth 생성, 20 epoch, clean acc 약 90.8%)
python train_model.py

# 검증기가 import할 수 있도록 모델 + 커스텀 로더 복사
cp fashion_mnist_cnn.pth alpha-beta-CROWN/complete_verifier/models/
cp fashion_mnist_model.py alpha-beta-CROWN/complete_verifier/custom/
```

### 2단계: MNIST FC 준비 (Marabou 비교용)

과제 #3의 ONNX는 가중치를 *external data*로 저장하며 맨 앞에 `Reshape` 노드가 있어
α,β-CROWN의 `onnx2pytorch` 변환이 멈춘다. ONNX를 직접 로드하는 대신, 가중치를 동일한
PyTorch 모듈로 이전하여 사용한다. (`mnist_fc.onnx`와 `mnist_fc.onnx.data`는 본 저장소에
이미 포함되어 있다.)

```bash
# 모델 정의 모듈을 검증기 custom 폴더로 복사
cp mnist_fc_model.py alpha-beta-CROWN/complete_verifier/custom/

# ONNX 가중치를 PyTorch로 이전 → 레포 루트에 mnist_fc.pth 생성
#   (onnxruntime 대비 max abs diff ≈ 6.7e-6 으로 일치 검증)
python convert_onnx_to_pth.py

# 생성된 .pth를 검증기가 로드하는 위치로 복사
cp mnist_fc.pth alpha-beta-CROWN/complete_verifier/models/
```

> `mnist_fc()`는 `net.` 접두사 없는 순수 `nn.Sequential`(키 `0./2./4.`)을 반환하므로,
> `convert_onnx_to_pth.py`는 ONNX 가중치를 이 키 형식에 맞춰 저장한다.

### 3단계: 전체 검증 실행

```bash
# 인자 없이 실행하면 configs/ 의 모든 YAML(FashionMNIST 7개 + MNIST FC 5개)을 순차 실행
CUDA_VISIBLE_DEVICES=0 python test.py

# 특정 config 하나만 실행하려면:
python test.py --config configs/mnist_fc_eps_0.05.yaml
```

각 config의 결과는 `results/<config_name>/` 폴더에 인스턴스별 상태
(verified / falsified / timeout)와 소요 시간을 담은 `results_summary.txt`,
그리고 원본 로그 `log_<config_name>.txt`로 저장된다.

### (선택) 보고서용 비교 figure 생성

두 검증기의 결과가 모두 준비되면(α,β-CROWN은 `results/`, Marabou는 과제 #3의
`summary.json`), 다음으로 비교 그래프를 생성한다:

```bash
python make_comparison_figs.py   # report_figs/ 에 PNG 저장
```

## 결과

전체 검증 결과(ε별 verified / falsified / timeout 분포, 검증 시간, 난이도 분석,
Marabou와의 비교)는 `report.pdf`에 정리되어 있다. config별 원시 결과는
`results/<config_name>/results_summary.txt`에서 확인할 수 있다.

## 비고

- `python test.py`는 인자 없이 실행하면 `configs/`의 모든 YAML을 순차 실행하고,
  `--config <yaml>`로 특정 config 하나만 실행할 수도 있다. 각 config의 결과는
  `results/<config_name>/`에 저장된다.
- 어려운 FashionMNIST config(ε ≥ 6/255)는 BaB 도메인의 메모리 폭증(OOM)을 막기 위해
  YAML 안에서 solver `batch_size: 512`로 설정되어 있다.
- 사용 하드웨어: RTX 6000 Ada (48 GB) GPU. `CUDA_VISIBLE_DEVICES`로 사용할 GPU를 지정한다.