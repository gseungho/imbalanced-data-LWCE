# image_detection — 장꼬리 객체 탐지 (Object Detection)

## 프로젝트 개요

**목적**: COCO/LVIS 객체 탐지 데이터셋에서 **LWCE 계열 손실함수**의 성능을 검증.
image_classification 이미지 분류 실험을 객체 탐지 도메인으로 확장하는 experiment.

**핵심 과제**: 
- Anchor-based detection에서 극심한 클래스 불균형 (배경 앵커 99:1 이상)
- Focal Loss로 배경/전경 불균형 완화 후, **LWCE+Focal** 조합으로 드물 클래스 개선
- 평가: mAP, per-class AP (특히 드물 클래스 Δ AP 비교)

**구조**: 
- MMDetection 프레임워크 (Faster-RCNN 베이스라인)
- image_classification의 손실함수 7종 적용 (ce, pwce, lwce, plwce, cb, focal)
- Optuna alpha/gamma 탐색 → 최종 평가
- COCO (80 classes, 표준) → LVIS (1203 classes, 극심 long-tail) 순차 검증

---

## 폴더 구조

```
image_detection/
├── custom_losses.py          # image_classification/custom_losses.py 재사용
│                             # (calculate_weights, FocalLoss, ClassificationLoss)
├── COCO.ipynb                # COCO 데이터셋 객체 탐지 실험 노트북
├── LVIS.ipynb                # LVIS 데이터셋 장꼬리 검증 노트북 (선택)
├── CLAUDE.md                 # 이 파일
└── results/
    ├── COCO/
    │   └── coco_results.json/.xlsx
    └── LVIS/
        └── lvis_results.json/.xlsx
```

---

## 손실함수 (7종)

image_classification/custom_losses.py의 손실함수를 객체 탐지 헤드에 적용.

### 분류 손실함수 비교

| loss_name | 가중치 모드 | 파라미터 | Optuna | 범위 |
|-----------|------------|---------|--------|------|
| `ce` | 균일 (없음) | — | × | — |
| `wce` | `total / n` (역빈도) | — | × | — |
| `pwce` | `(total/n)^alpha` | alpha | ○ | [0.5, 5.0] |
| `lwce` | `1 / log1p(n)` | — | × | — |
| `plwce` | `1 / log1p(n)^alpha` | alpha | ○ | [2.0, 15.0] |
| `cb` | `(1-beta) / (1-beta^n)` | beta=0.9999 | × | — |
| `focal` | CE + `(1-pt)^gamma` | gamma | ○ | [0.5, 5.0] |

**가중치 파싱 우선순위** (substring 충돌 방지):
```
plwce > lwce > pwce > wce > cb > ce
```

**적용 전략**:
- RPN 분류 손실: Focal Loss (배경/전경 99:1 불균형 완화)
- 헤드 분류 손실: 선택된 loss_name (클래스 불균형 완화)
- 각 손실은 독립적으로 적용 (합산 가능)

### ClassificationLoss 클래스

```python
from custom_losses import get_clf_loss

# COCO: 80 classes, class_counts[각 클래스의 이미지 수]
loss = get_clf_loss('plwce', class_counts, alpha=2.0, gamma=2.0)
logits = model(images)  # (B, 80) shape
class_targets = ...     # (B,) 
loss_val = loss(logits, class_targets)
```

---

## 데이터셋 (장꼬리 분포)

### COCO 2017
- **클래스 수**: 80 (person, car, dog, ... 등)
- **이미지 수**: train 118K, val 5K
- **불균합 비율**: ~5~10:1 (head: person ~30K img, tail: toothbrush ~200 img)
- **앵커 불균합**: 배경:전경 = 99:1 이상 (Anchor-based detection 특성)
- **다운로드**: `torchvision.datasets.CocoDetection` 또는 official COCO API

### LVIS v1.0 (선택 과제)
- **클래스 수**: 1203 (매우 세분화)
- **이미지 수**: 100K
- **불균합 비율**: 극심 long-tail (head: rare 클래스 ~1 img, common 클래스 ~10K img)
- **목적**: LWCE 효과 극대화 검증

---

## 모델 아키텍처

### Faster-RCNN (MMDetection)

```
Backbone: ResNet-50 (ImageNet pretrained)
    ↓
FPN: Feature Pyramid Network (P3~P7)
    ↓
RPN: Region Proposal Network
    ├─ cls_loss (배경:전경 = 99:1 → Focal Loss 필수)
    └─ reg_loss (L1/smooth L1)
    ↓
RoI Head (Detector)
    ├─ cls_loss (클래스 불균합 → 손실함수 선택 대상)
    ├─ reg_loss
    ↓
Output: 80 class scores + bbox offsets
```

**설치**:
```bash
pip install mmdetection mmengine
```

**팩토리 함수**:
```python
def build_detector(num_classes, pretrained=True):
    """Faster-RCNN + ResNet-50 + FPN"""
    return ... # MMDetection config 기반
```

---

## 노트북 구조 (8셀)

### Cell 0: 환경설정

```python
!pip install -q torch torchvision mmdetection mmengine openpyxl pandas scikit-image

import os, sys, json, numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 상수
DATASET_NAME = 'COCO'
NUM_CLASSES = 80
BATCH_SIZE = 16
NUM_WORKERS = 0
EPOCHS = 50
RESULTS_DIR = f'./results/{DATASET_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)
```

### Cell 1: 데이터 로드

```python
def load_coco_data(batch_size, num_workers=0):
    """
    COCO 2017 데이터셋 로드 및 분할
    
    반환:
        train_loader, val_loader, test_loader, class_counts
    """
    # torchvision.datasets.CocoDetection로 로드
    # class_counts: 각 클래스의 이미지 수 계산
    # train_loader, val_loader, test_loader 생성
    pass

train_loader, val_loader, test_loader, class_counts = load_coco_data(
    BATCH_SIZE, NUM_WORKERS
)
print(f"Class distribution: {class_counts}")
```

### Cell 2: 클래스 분포 시각화

```python
def visualize_class_distribution(class_counts, dataset_name='COCO'):
    """
    막대 그래프 (log scale), Head/Mid/Tail 그룹 경계 표시
    """
    pass

visualize_class_distribution(class_counts, DATASET_NAME)
```

### Cell 3: 모델 및 평가 함수 정의

```python
def build_detector(num_classes):
    """
    Faster-RCNN (ResNet-50 + FPN) 반환
    """
    pass

def compute_val_metrics(model, loader, num_classes, class_counts=None):
    """
    객체 탐지 평가 지표 계산
    
    반환: {
        'mAP': float,              # 모든 클래스 평균 AP @ IoU=0.5:0.95
        'AP50': float,             # AP @ IoU=0.5
        'Per_Class_AP': [ap0, ap1, ...],  # 각 클래스 AP
        'Head_AP': float,          # 많은 클래스 AP 평균
        'Mid_AP': float,           # 중간 클래스 AP 평균
        'Tail_AP': float,          # 드문 클래스 AP 평균
        ...
    }
    """
    pass
```

### Cell 4: train_model() 함수 정의 (실행 안 함)

```python
def train_model(loss_name, class_counts, train_loader, val_loader, num_classes,
                alpha=1.0, gamma=2.0, epochs=50, tag=''):
    """
    모델 학습
    
    Args:
        loss_name: 'ce', 'pwce', 'lwce', 'plwce', 'cb', 'focal'
        class_counts: 클래스별 이미지 수
        alpha: PLWCE/PWCE 파라미터
        gamma: Focal Loss 파라미터
        epochs: 학습 에포크
    
    반환: (model, history, best_mAP)
    """
    pass
```

### Cell 5: Optuna 탐색 (proxy 기반)

```python
os.environ['TQDM_DISABLE'] = '1'

PROXY_EPOCHS = 15
PROXY_SUBSET_RATIO = 0.10    # 전체 train 데이터의 10% 사용
N_TRIALS_PWCE = 15
N_TRIALS_PLWCE = 15
N_TRIALS_FOCAL = 15
ALPHA_LOW, ALPHA_HIGH = 0.5, 5.0
GAMMA_LOW, GAMMA_HIGH = 0.5, 5.0

optuna_best = {}

# GridSampler 사용, MedianPruner 미사용
# plwce alpha 탐색, focal gamma 탐색
```

### Cell 6: 전체 실험 실행

```python
LOSS_CONFIGS = ['ce', 'pwce', 'lwce', 'plwce', 'cb', 'focal']

all_results = {}
all_histories = {}

for loss_name in LOSS_CONFIGS:
    alpha = optuna_best.get(loss_name, {}).get('alpha', 1.0)
    gamma = optuna_best.get('focal', {}).get('gamma', 2.0) if loss_name == 'focal' else 2.0
    
    model, history, best_mAP = train_model(
        loss_name, class_counts, train_loader, val_loader,
        NUM_CLASSES, alpha=alpha, gamma=gamma, epochs=EPOCHS
    )
    metrics = compute_val_metrics(model, test_loader, NUM_CLASSES, class_counts)
    
    all_results[loss_name] = metrics
    all_histories[loss_name] = history
    print(f"{loss_name}: mAP={metrics['mAP']:.4f}")
```

### Cell 7: 평가, 시각화, 결과 저장

```python
# 손실함수별 mAP/AP50/Tail_AP 시각화
# JSON 저장: f"{RESULTS_DIR}/coco_results.json"
# Excel 저장: f"{RESULTS_DIR}/coco_results.xlsx"
#   - Summary sheet: 손실함수별 mAP, AP50, Head/Mid/Tail AP
#   - Per_Class_AP sheet: 각 클래스별 AP
#   - Training_History sheet: loss_name별 epoch 추이
```

---

## 평가 지표

### 객체 탐지 메트릭

| 지표 | 정의 |
|------|------|
| **mAP** | 모든 클래스 평균 Precision-Recall곡선 아래 면적 (IoU=0.5:0.95) |
| **AP50** | AP @ IoU threshold = 0.5 |
| **AP75** | AP @ IoU threshold = 0.75 |
| **Per-Class AP** | 각 클래스별 AP |

### 그룹별 메트릭 (class_counts 기반)

```
Head_AP    = 이미지 많은 클래스 (n_i ≥ threshold1) AP 평균
Mid_AP     = threshold1 > n_i ≥ threshold2 클래스 AP 평균
Tail_AP    = 드문 클래스 (n_i < threshold2) AP 평균
```

**그룹 경계**: 실제 `class_counts`의 quantile에서 동적 계산 (COCO는 ~5~10:1, LVIS는 극심).

---

## Optuna 설정

### 파라미터 탐색 범위

#### pwce (alpha만 탐색)
```python
N_TRIALS_PWCE = 15
alphas = linspace(0.5, 5.0, 15)
GridSampler({'alpha': alphas})
```

#### plwce (alpha만 탐색, 더 높은 범위)
```python
N_TRIALS_PLWCE = 15
alphas = linspace(2.0, 15.0, 15)  # 더 강한 가중치 범위
GridSampler({'alpha': alphas})
```

#### focal (gamma만 탐색)
```python
N_TRIALS_FOCAL = 15
gammas = linspace(0.5, 5.0, 15)
GridSampler({'gamma': gammas})
```

### 주의사항
- **MedianPruner 미사용** — GridSampler와 병행 금지
- **proxy 메트릭**: `mAP` (객체 탐지 표준 지표)
- **Optuna 출력 억제**: `os.environ['TQDM_DISABLE'] = '1'`

---

## 알려진 버그 패턴 (image_classification과 동일)

| 항목 | 잘못된 방식 | 올바른 방식 |
|------|-----------|----------|
| Trial 필터 | `if t.value:` | `if t.value is not None:` |
| study 시각화 순서 | study 정의 전 plot 코드 | study 정의 후 plot 코드 |
| GridSampler + Pruner | 병행 사용 | GridSampler 단독 |
| `NUM_WORKERS` | 2 이상 | 0 고정 (Colab notebook) |

---

## 결과 저장 위치 및 형식

### 경로 구조
```
results/
├── COCO/
│   ├── coco_results.json
│   └── coco_results.xlsx
└── LVIS/
    ├── lvis_results.json
    └── lvis_results.xlsx
```

### JSON 포맷
```json
{
  "ce": {
    "mAP": 0.382,
    "AP50": 0.586,
    "AP75": 0.415,
    "Per_Class_AP": [0.45, 0.52, ..., 0.12],
    "Head_AP": 0.42,
    "Mid_AP": 0.38,
    "Tail_AP": 0.28
  },
  "pwce": {...},
  "plwce": {...},
  ...
}
```

### Excel 시트 구성
- **Summary**: 손실함수별 mAP, AP50, AP75, Head/Mid/Tail AP
- **Per_Class_AP**: 각 클래스별 AP 비교
- **Training_History**: (loss_name, epoch, train_loss, val_mAP, ...) 추이

---

## 구현 순서

1. **COCO.ipynb** (우선)
   - Faster-RCNN 베이스라인 확립
   - 손실함수 7종 비교 실행
   - mAP 성능 기록

2. **LVIS.ipynb** (확장)
   - 극심 장꼬리 데이터셋에서 LWCE 효과 검증
   - Tail AP 개선 효과 정량화

---

## 실행 환경

**Colab 추천 GPU**: A100 (40GB VRAM) 또는 V100 (32GB VRAM)
- Faster-RCNN 학습: ~30~40 min/epoch (COCO 18K image subset)
- 전체 실험 (7 손실함수 × 50 epochs): ~20 시간

**로컬 실행**: GPU 메모리 ≥ 16GB

---

## 예상 결과

근거: image_classification 실험에서
- 불균형 ≤ 5:1 → LWCE 효과 미미
- 불균형 > 50:1 → plwce > lwce ≥ ce

**COCO (불균합 ~5~10:1)**:
- `focal` → Focal Loss (배경 제거 후 클래스 불균합 >90%)
- `plwce` + Focal → Δ mAP +0.5~1.0% 예상 (tail class AP 개선)

**LVIS (극심 불균합, 1000:1+)**:
- `plwce` 효과 극대화 예상
- Tail AP: `ce` 대비 `plwce` +3~5% 향상 가능

---

## 참고 자료

- image_classification/CLAUDE.md — 이미지 분류 실험 설정
- image_classification/custom_losses.py — 손실함수 구현 (7종)
- MMDetection docs: https://mmdetection.readthedocs.io
- COCO API: https://github.com/cocodataset/cocoapi
- LVIS API: https://github.com/lvis-dataset/lvis-api
