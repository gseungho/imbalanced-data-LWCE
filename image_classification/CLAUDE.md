# image_classification — CIFAR-10/100 Long-Tail 이미지 분류 실험

## 프로젝트 개요

**목적**: CIFAR-10/100 Long-Tail(LT) 불균형 데이터셋에서 **LWCE 계열 손실함수**의 분류 성능을 검증.
medical_data 세그멘테이션 실험을 이미지 분류 도메인으로 확장하는 experiment.

**구조**: 
- 인공 불균형 생성 (지수 감소 분포, IR=10/50/100)
- 6가지 손실함수 비교 (ce, pwce, lwce, plwce, cb, focal)
- Optuna hyperparameter 탐색 (pwce/plwce alpha, focal gamma)
- ResNet-32 backbone (CIFAR 전용 아키텍처)

---

## 폴더 구조

```
image_classification/
├── custom_losses.py       # 분류용 손실함수 (calculate_weights, FocalLoss)
├── resnet32.py            # ResNet-32 CIFAR 구현 (3 stage × 6 blocks = 32층)
├── CIFAR10_LT.ipynb       # CIFAR-10 LT 실험 노트북
├── CIFAR100_LT.ipynb      # CIFAR-100 LT 실험 노트북
├── CLAUDE.md              # 이 파일
└── results/
    ├── CIFAR10_LT/
    │   ├── IR10/          # cifar10_lt_ir10_results.json/.xlsx
    │   ├── IR50/
    │   └── IR100/
    └── CIFAR100_LT/
        ├── IR10/
        ├── IR50/
        └── IR100/
```

---

## 손실함수 (6종)

`custom_losses.py`에서 제공하는 6가지 손실함수:

| loss_name | 가중치 방식 | 파라미터 | Optuna | 범위 |
|-----------|-----------|---------|--------|------|
| `ce` | 균일 (가중치 없음) | — | × | — |
| `pwce` | `(total / n)^alpha` | alpha | ○ | [0.3, 5.0] |
| `lwce` | `1 / log1p(n)` | — | × | — |
| `plwce` | `1 / log1p(n)^alpha` | alpha | ○ | [0.5, 6.0] |
| `cb` | `(1-beta) / (1-beta^n+1e-6)` | beta=0.9999 | × | — |
| `focal` | CE + `(1-pt)^gamma` | gamma | ○ | [1.0, 5.0] |

**가중치 파싱 우선순위** (substring 충돌 방지):
```
plwce > lwce > pwce > wce > cb > ce
```

**ClassificationLoss 클래스**:
- `__init__(loss_name, class_counts, alpha=1.0, beta=0.9999, gamma=2.0)`
- loss_name 문자열에서 가중치 모드 결정
- Focal Loss 또는 CrossEntropyLoss 동적 선택
- Device 동기화 (Colab GPU 불일치 대비)

---

## ResNet-32 아키텍처

He et al. 2016 CIFAR 변형. 32×32 입력 최적화.

```
Input (B, 3, 32, 32)
    ↓
conv1: 3→16 channels, 3×3, stride=1, no maxpool
    ↓
layer1: 6 × BasicBlock(16, 16, stride=1)     → (B, 16, 32, 32)
    ↓
layer2: 6 × BasicBlock(16→32, stride=2)      → (B, 32, 16, 16)
    ↓
layer3: 6 × BasicBlock(32→64, stride=2)      → (B, 64, 8, 8)
    ↓
GlobalAvgPool2d
    ↓
Linear(64, num_classes)
    ↓
Output logits
```

**사양**:
- 총 32 weight layer (1 + 3×(6×2) + 1)
- ~0.47M 파라미터 (CIFAR-10), ~0.48M (CIFAR-100)
- Kaiming 초기화

**팩토리 함수**: `build_resnet32(num_classes)`

---

## 데이터셋 (Long-Tail 생성)

### 불균형 생성 공식
```
n_i = n_max × IR^(-i/(K-1))
```

### CIFAR-10 LT
- `n_max = 5000` (클래스당 최대 샘플 수)
- `K = 10` (클래스 개수)
- IR=100 일 때: `n_min = 50` (클래스당 최소 샘플 수)

### CIFAR-100 LT
- `n_max = 500`
- `K = 100`
- IR=100 일 때: `n_min = 5`

### Split 비율
```
Train : Val : Test = 8 : 1 : 1   (stratified)
```

### 실험 IR 목록
```
IR_LIST = [10, 50, 100]
```

---

## 학습 설정

### 하이퍼파라미터
```python
BATCH_SIZE   = 128
FINAL_EPOCHS = 200
NUM_WORKERS  = 0        # Colab notebook 필수 (worker cleanup 오류 방지)
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 최적화기
```python
optimizer = SGD(lr=0.1, momentum=0.9, weight_decay=2e-4)
scheduler = MultiStepLR(milestones=[160, 180], gamma=0.01)
```

### Data Augmentation
```
RandomCrop(32, padding=4)
RandomHorizontalFlip(p=0.5)
Normalize(mean=[0.4914, 0.4822, 0.4465],
          std=[0.2023, 0.1994, 0.2010])
```

---

## Optuna 하이퍼파라미터 탐색

### 설정
```python
PROXY_EPOCHS        = 20
PROXY_SUBSET_RATIO  = 0.20    # 전체 train 데이터의 20% 사용
PROXY_MIN_PER_CLASS = 5       # stratified proxy: 클래스별 최소 보장 (tail 소멸 방지)
```

> **stratified proxy (v2)**: 기존 `np.random.choice` 랜덤 추출은 CIFAR-100처럼
> 클래스가 많고 tail 샘플(4~5개)이 희박하면 proxy에서 tail 클래스가 사라져
> (P(0개)≈41%) Optuna가 항상 가장 약한 α(≈CE)를 선택하는 문제가 있었음.
> v2는 클래스별 `max(0.2·n_c, min(n_c, 5))`개를 stratified 추출해 tail을 보존함.
> 체크포인트는 `*_v2.json`으로 분리 (v1 결과와 독립).

### GridSampler 구성 (탐색 범위는 tabular/network 노트북과 통일)

#### pwce (alpha)
```
N_TRIALS_PWCE = 20
alphas = linspace(0.3, 5.0, 20)   # α<1 (약한 가중치)부터 탐색
GridSampler({'alpha': alphas})
```

#### plwce (alpha)
```
N_TRIALS_PLWCE = 20
alphas = linspace(0.5, 6.0, 20)   # α=0.5(LWCE보다 약함) ~ 6.0(공격적)
GridSampler({'alpha': alphas})
```

#### focal (gamma)
```
N_TRIALS_FOCAL = 20
gammas = linspace(1.0, 5.0, 20)   # γ≥1 — γ<1은 NaN 붕괴 (아래 버그표 참조)
GridSampler({'gamma': gammas})
```

### 참고사항
- **MedianPruner 미사용** — GridSampler와 병행 금지
- **n_trials 정확성** — trial 수가 grid 크기와 정확히 일치해야 함
- **Proxy metric**: `balanced_acc` (class-wise recall 평균)
  - Top-1 Acc 대신 사용 — Few-shot 클래스 반영 필요
- **Optuna 출력 억제**: `os.environ['TQDM_DISABLE'] = '1'`

---

## 평가 지표

### 주요 지표
| 지표 | 정의 |
|------|------|
| **Top-1 Acc** | 전체 정확도 |
| **Balanced Acc** | 클래스별 recall의 macro 평균 (few-shot 클래스 가중) |
| **F1-Macro** | sklearn.metrics.f1_score(..., average='macro', zero_division=0) |

### 그룹별 정확도
```
Many-shot Acc  = n_i ≥ 100 클래스의 정확도 평균
Medium-shot Acc = 20 ≤ n_i < 100 클래스의 정확도 평균
Few-shot Acc   = n_i < 20 클래스의 정확도 평균
```

**그룹 경계**: 실제 `class_counts`에서 동적 계산 (IR마다 다름)

### Per-Class Accuracy
```
per_class_acc = [정확도_클래스0, 정확도_클래스1, ..., 정확도_클래스K-1]
```

---

## 노트북 구조

### Cell 0: 환경설정
- pip install (optuna, torchvision, pandas, openpyxl)
- import, device, 경로 설정
- 상수: DATASET, NUM_CLASSES, IR_LIST, BATCH_SIZE, etc.

### Cell 1: 데이터 로드
- `make_cifar_lt(dataset, ir)` — torchvision 다운로드 + 지수 감소 서브샘플링
- 반환: `(train_loader, val_loader, test_loader, class_counts)`

### Cell 2: 클래스 분포 시각화
- 막대 그래프 (log scale)
- Many/Medium/Few 그룹 경계 출력
- 샘플 수 통계

### Cell 3: 모델 및 평가 함수
- `build_model()` — ResNet-32 인스턴스화
- `compute_val_acc(model, loader)` — Top-1 정확도
- `compute_val_metrics(model, loader, num_classes, class_counts)` — 모든 지표

### Cell 4: 학습 함수
- `train_model(loss_name, class_counts, train_loader, val_loader, ...)` 정의
- 반환: `(model, history, best_val_acc)`
- **실행하지 않음** (Optuna cell에서만 호출)

### Cell 5: Optuna 탐색
- 각 IR별로 pwce/plwce/focal 3가지 search 실행
- `optuna_best[ir]` 딕셔너리에 최적값 저장
  ```python
  optuna_best[ir] = {
      'pwce': {'alpha': ...},
      'plwce': {'alpha': ...},
      'focal': {'gamma': ...},
  }
  ```

### Cell 6: 전체 실험
```python
LOSS_CONFIGS = ['ce', 'pwce', 'lwce', 'plwce', 'cb', 'focal']

for ir in IR_LIST:
    for loss_name in LOSS_CONFIGS:
        alpha = optuna_best[ir].get(loss_name, {}).get('alpha', 1.0)
        gamma = optuna_best[ir].get('focal', {}).get('gamma', 2.0) if loss_name=='focal' else 2.0
        
        model, history, _ = train_model(loss_name, class_counts, train_loader, val_loader,
                                         ..., alpha=alpha, gamma=gamma, ...)
        metrics = compute_val_metrics(model, test_loader, ...)
        all_results[ir][loss_name] = metrics
        all_histories[ir][loss_name] = history
```

### Cell 7: 평가 및 저장
- 시각화 (손실함수 × IR 비교 그래프)
- JSON 저장: `results/CIFAR{10,100}_LT/IR{ir}/cifar{10,100}_lt_ir{ir}_results.json`
- Excel 저장:
  - `Summary` sheet: 손실함수별 전체 지표
  - `Training_History` sheet: (loss_name, epoch, train_loss, val_acc, ...)
  - `Per_Class_Acc` sheet: 손실함수별 클래스별 정확도

---

## 알려진 버그 및 주의사항

### Optuna 관련
| 항목 | 잘못된 방식 | 올바른 방식 |
|------|-----------|----------|
| Trial 필터 | `if t.value:` | `if t.value is not None:` |
| study 시각화 순서 | study 정의 전 plot 코드 | study 정의 후 plot 코드 |
| Pruner | GridSampler + MedianPruner 병행 | GridSampler 단독 사용 |
| proxy 서브셋 | `np.random.choice` 랜덤 (tail 클래스 소멸 → α≈CE로 degenerate) | 클래스별 `min(n_c, 5)` 보장 stratified 추출 |
| focal gamma 하한 | `linspace(0.5, ...)` (γ<1) | `linspace(1.0, ...)` — γ<1은 NaN 붕괴 |

> **Focal γ<1 NaN 붕괴**: softmax focal의 `(1−pt)^γ` gradient `−γ(1−pt)^(γ−1)`는 γ<1일 때
> `pt→1`에서 발산 → 다수 클래스가 confident해지면 gradient 폭발 → NaN → 전 샘플 class 0 예측
> (`argmax(NaN)=0`). 증상: focal F1≈`F1_class0/K`, std=0, Few/Tail=0. Optuna proxy(짧은 학습)는
> γ<1을 잘못 "최적"으로 뽑으므로, grid 하한을 반드시 `1.0`으로 둘 것 (γ=0은 CE라 안전, 0<γ<1만 위험).
> Network 통합 노트북에서 2026-06 확인. medical_data의 `plwce_focal_dice`는 Dice 결합이라 별개.

### 학습 관련
| 항목 | 주의 |
|------|------|
| `NUM_WORKERS` | 반드시 0으로 고정 (Colab notebook) |
| Loss device 동기화 | `ClassificationLoss.forward()`에서 자동 처리 |
| 초기 모델 호출 | Cell 정의 후 호출하지 말 것 (불필요한 가중치 다운로드) |

---

## 결과 저장 위치 및 형식

### 경로 구조
```
results/
├── CIFAR10_LT/
│   ├── IR10/
│   │   └── cifar10_lt_ir10_results.json
│   │   └── cifar10_lt_ir10_results.xlsx
│   ├── IR50/
│   └── IR100/
└── CIFAR100_LT/
    ├── IR10/
    ├── IR50/
    └── IR100/
```

### JSON 포맷
```json
{
  "ce": {
    "Top1_Acc": 0.73,
    "Balanced_Acc": 0.65,
    "F1_Macro": 0.64,
    "Many_Acc": 0.81,
    "Medium_Acc": 0.68,
    "Few_Acc": 0.42,
    "Per_Class_Acc": [0.90, 0.85, ..., 0.10]
  },
  "pwce": {...},
  ...
}
```

### Excel 시트 구성
- **Summary**: 손실함수별 Top-1, Balanced Acc, F1-Macro, Many/Medium/Few 정확도
- **Training_History**: (loss_name, epoch, train_loss, val_top1_acc, val_balanced_acc)
- **Per_Class_Acc**: 손실함수별 클래스 0~K-1의 정확도

---

## 실험 결과 (5 seeds, mean ± std, F1-Macro 기준)

그룹 정의: training count 기준 상위/중위/하위 1/3 (tertile split)

### CIFAR-10-LT

| Loss  | IR=10 F1 | IR=50 F1 | IR=100 F1 | IR=100 Few |
|-------|----------|----------|-----------|------------|
| ce    | 0.7103±0.0074 | 0.5203±0.0140 | 0.4306±0.0209 | 0.2094 |
| pwce  | 0.7235±0.0092 | 0.5652±0.0058 | 0.4564±0.0233 | 0.2394 |
| lwce  | 0.7182±0.0100 | 0.5612±0.0227 | 0.4459±0.0159 | 0.2222 |
| plwce | 0.7268±0.0154 | 0.5551±0.0209 | 0.4583±0.0175 | 0.2347 |
| cb    | 0.7296±0.0104 | 0.5512±0.0167 | 0.4063±0.0150 | 0.2044 |
| focal | 0.7143±0.0160 | 0.5137±0.0279 | 0.4565±0.0135 | 0.2336 |

### CIFAR-100-LT

| Loss  | IR=10 F1 | IR=50 F1 | IR=100 F1 | IR=100 Few |
|-------|----------|----------|-----------|------------|
| ce    | 0.3513±0.0175 | 0.2101±0.0058 | 0.1786±0.0033 | 0.0431 |
| pwce  | 0.3652±0.0073 | 0.2107±0.0061 | 0.1783±0.0045 | 0.0449 |
| lwce  | 0.3590±0.0074 | 0.2173±0.0076 | 0.1846±0.0051 | 0.0451 |
| plwce | 0.3582±0.0246 | 0.2186±0.0054 | 0.1859±0.0088 | 0.0506 |
| cb    | 0.3598±0.0155 | 0.1960±0.0051 | 0.1577±0.0041 | 0.0373 |
| focal | 0.3625±0.0159 | 0.2084±0.0062 | 0.1689±0.0043 | 0.0380 |

### 주요 소견

- **IR=10**: 차이 미미. CB가 CIFAR-10에서 근소 우위, CIFAR-100에서는 pwce/focal 상위
- **IR=50**: CIFAR-10에서 pwce > lwce > CE 명확. CIFAR-100에서 plwce > lwce > CE
- **IR=100**: LWCE/PLWCE가 CE 대비 일관되게 우위. CB는 두 데이터셋 모두 최하위
- **Few-shot**: IR 증가할수록 LWCE/PLWCE의 Few 클래스 이점 뚜렷 (특히 CIFAR-100)
- **PWCE**: CIFAR-10에서 plwce와 경쟁하나 CIFAR-100 IR=100에서 CE와 동등 수준으로 하락
- **Focal**: IR=100 CIFAR-10에서 plwce와 경쟁하지만 IR=50에서 CE보다 낮아 불안정

### Optuna 최적 파라미터

| 데이터셋 | IR | pwce α | plwce α | focal γ |
|----------|----|--------|---------|---------|
| CIFAR-10 | 10  | 0.765 | 5.941 | 2.158 |
| CIFAR-10 | 50  | 0.500 | 2.647 | 1.211 |
| CIFAR-10 | 100 | 0.500 | 2.647 | 1.211 |
| CIFAR-100 | 10  | 0.500 | 1.000 | 4.526 |
| CIFAR-100 | 50  | 0.500 | 1.000 | 1.447 |
| CIFAR-100 | 100 | 0.500 | 1.000 | 4.053 |

---

## 참고 링크

- medical_data/CLAUDE.md — 의료 세그멘테이션 실험 문서
- medical_data/rules.md — 손실함수 설계 및 SoTA 비교
- https://github.com/kaidic/LDAM-DRW — Long-Tail Learning 벤치마크
