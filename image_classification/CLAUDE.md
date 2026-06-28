# image_classification — CIFAR-10/100 Long-Tail 이미지 분류 실험

## 프로젝트 개요

**목적**: CIFAR-10/100 Long-Tail(LT) 불균형 데이터셋에서 **LWCE 계열 손실함수**의 분류 성능을 검증.
medical_data 세그멘테이션 실험을 이미지 분류 도메인으로 확장하는 experiment.

**구조**: 
- 인공 불균형 생성 (지수 감소 분포, IR=10/50/100)
- 8가지 손실함수 비교 (ce, wce, pwce, sqce, lwce, plwce, cb, focal)
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

## 손실함수 (8종)

`custom_losses.py`에서 제공하는 8가지 손실함수:

| loss_name | 가중치 방식 | 파라미터 | Optuna | 범위 | 역할 |
|-----------|-----------|---------|--------|------|------|
| `ce` | 균일 (가중치 없음) | — | × | — | baseline |
| `wce` | `total / n` (역빈도) | — | × | — | baseline (`= pwce α=1`) |
| `pwce` | `(total / n)^alpha` | alpha | ○ | [0.3, 5.0] | 분석/이론 (α-sweep foil, main 비교 아님) |
| `sqce` | `√(total / n)` (역제곱근 빈도, ∝ 1/√n) | — | × | — | reported baseline |
| `lwce` | `1 / log1p(n)` | — | × | — | **proposed** |
| `plwce` | `1 / log1p(n)^alpha` | alpha | ○ | [0.5, 6.0] | **proposed** |
| `cb` | `(1-beta) / (1-beta^n+1e-6)` | beta=0.9999 | × | — | baseline |
| `focal` | CE + `(1-pt)^gamma` | gamma | ○ | [1.0, 5.0] | baseline |

**가중치 파싱 우선순위** (substring 충돌 방지):
```
plwce > lwce > pwce > sqce > wce > cb > ce
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
LOSS_CONFIGS = ['ce', 'wce', 'pwce', 'sqce', 'lwce', 'plwce', 'cb', 'focal']  # wce=역빈도(=pwce α=1), sqce=√-CE(1/√n) α=0.5 고정

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
수치 출처: `results_checkpoint_cifar{10,100}_v2.json` (2026-06 재실행, focal γ≥1.0 하한 + stratified proxy v2, sqce 포함). **lwce/plwce가 proposed.**
**wce 추가 (2026-06, `wce=pwce α=1`, ✅ CIFAR-10·100 모두 완료)**: wce가 **IR 의존적으로 붕괴** — image LT에서 가장 깨끗하게 보임.
- **CIFAR-10 (가장 극적)**: wce가 **IR=10 rank 1/8(전체 최우수 F1 0.7268, Few-acc도 1위) → IR=50 5/8 → IR=100 8/8(꼴찌, F1 0.3935·Few 0.1850 최하)**. 완벽 단조 best→worst = weight explosion 교과서 시연.
- **CIFAR-100**: IR=10 rank 5/8(Few-acc 0.2472 최고) → IR=50·100 7/8.
- network/tabular(거의 항상 꼴찌)와 대조: image LT는 **저IR 경쟁력→고IR 붕괴**의 IR 의존성. α-sweep(image LT는 가중 레버리지 큼)과 정합.

### CIFAR-10-LT

| Loss  | IR=10 F1 | IR=50 F1 | IR=100 F1 | IR=100 Few |
|-------|----------|----------|-----------|------------|
| ce    | 0.7058±0.0057 | 0.5026±0.0285 | 0.4282±0.0318 | 0.2018 |
| wce   | **0.7268±0.0138** | 0.5404±0.0222 | 0.3935±0.0410 | 0.1850 |
| pwce  | 0.7131±0.0099 | 0.5638±0.0369 | 0.4711±0.0194 | 0.2576 |
| sqce  | 0.7201±0.0120 | 0.5677±0.0184 | 0.4772±0.0129 | 0.2634 |
| **lwce** | 0.7139±0.0035 | 0.5468±0.0192 | 0.4498±0.0135 | 0.2264 |
| **plwce** | 0.7134±0.0078 | 0.5586±0.0240 | 0.4635±0.0166 | 0.2508 |
| cb    | 0.7260±0.0169 | 0.5296±0.0397 | 0.4062±0.0391 | 0.2070 |
| focal | 0.7023±0.0080 | 0.5211±0.0305 | 0.4437±0.0318 | 0.2149 |

### CIFAR-100-LT

| Loss  | IR=10 F1 | IR=50 F1 | IR=100 F1 | IR=100 Few |
|-------|----------|----------|-----------|------------|
| ce    | 0.3466±0.0092 | 0.2185±0.0069 | 0.1818±0.0013 | 0.0432 |
| wce   | 0.3557±0.0216 | 0.1935±0.0030 | 0.1580±0.0054 | 0.0421 |
| pwce  | 0.3589±0.0157 | 0.2179±0.0060 | 0.1778±0.0072 | 0.0441 |
| sqce  | 0.3585±0.0093 | 0.2110±0.0054 | 0.1709±0.0025 | 0.0361 |
| **lwce** | 0.3637±0.0142 | 0.2170±0.0055 | 0.1781±0.0057 | 0.0464 |
| **plwce** | 0.3589±0.0036 | 0.2126±0.0053 | 0.1869±0.0097 | 0.0495 |
| cb    | 0.3463±0.0221 | 0.1876±0.0047 | 0.1563±0.0078 | 0.0394 |
| focal | 0.3433±0.0209 | 0.2203±0.0062 | 0.1818±0.0034 | 0.0457 |

### 주요 소견 (proposed = lwce/plwce)

- **CIFAR-100(100클래스)에서 제안 손실 강세**: **lwce가 IR=10 단독 1위**(0.3637), **plwce가 IR=100 단독 1위**(0.1869, Few도 0.0495 최상). 클래스 수가 많을수록 LWCE 계열 이점 부각 — tabular 다중 클래스(glass/steel/page_blocks)·network 다클래스와 동일 경향.
- **CIFAR-10(10클래스)**: IR=50/100에서 reweighting 전반(sqce/pwce/plwce/lwce)이 CE를 뚜렷이 상회. 제안 손실은 상위권 내 중위 — plwce가 pwce 다음(IR=50 0.5586 / IR=100 0.4635), lwce는 그 아래.
- **IR=10**: 전 손실 차이 미미(노이즈). CIFAR-10 cb 근소 1위, CIFAR-100 lwce 근소 1위.
- **고IR 최하위 = cb 또는 wce**: IR↑일수록 다수클래스 과억제 손실이 최하위. CIFAR-100 IR=100은 cb(0.1563) 최하, **CIFAR-10 IR=100은 wce(0.3935)가 cb(0.4062)를 제치고 단독 최하** — wce는 IR=10 최우수였다가 폭발(위 wce 주석 참조).
- **PWCE**: CIFAR-10 강세지만 CIFAR-100 IR=100에서 CE 수준(0.1778)으로 하락 — 고불균형·다클래스 불안정.
- **Focal**: γ≥1 수정 후 collapse 없음(std≠0, Few≠0). 중간 IR에서만 CE 상회.

> ⚠️ **sqce(√-CE) baseline 경쟁력 — 논문 대응 필요**: 파라미터 없는 sqce가 **CIFAR-10 IR=50(0.5677)·IR=100(0.4772)에서 제안 lwce/plwce를 앞섬**. 단 **CIFAR-100(다클래스)에선 sqce가 하위권**으로 떨어지고 제안 손실이 1위. network도 동일(few-class 극단 불균형 CIC-DDoS2019는 sqce, 다클래스 NSL-KDD/CICIDS2017은 lwce/plwce). → **"제안 손실은 다클래스·중간 불균형에 강하고, √-CE는 few-class 극단 불균형에 강하다"**는 역할 구분으로 프레이밍 권장.

### Optuna 최적 파라미터 (checkpoint 내장값, seed 불변)

| 데이터셋 | IR | pwce α | plwce α | focal γ |
|----------|----|--------|---------|---------|
| CIFAR-10 | 10  | 0.300 | 1.368 | 1.211 |
| CIFAR-10 | 50  | 0.547 | 3.105 | 2.684 |
| CIFAR-10 | 100 | 0.547 | 2.526 | 2.053 |
| CIFAR-100 | 10  | 0.547 | 3.105 | 1.000 |
| CIFAR-100 | 50  | 0.300 | 0.789 | 1.421 |
| CIFAR-100 | 100 | 0.300 | 0.500 | 1.000 |

---

## α-sweep 분석 — PLWCE α* vs IR (CIFAR-LT, 분석/이론 섹션용)

CIFAR-10-LT에서 IR을 고정 grid(10/20/50/100/200)로 두고 **각 IR마다 PLWCE α를 스윕**해 test-F1 peak 지점 α\*를 측정한 controlled 실험. (CIFAR-LT는 같은 데이터에서 IR만 변경 가능 → 교란변수 통제됨.)

| IR | 10 | 20 | 50 | 100 | 200 |
|----|----|----|----|----|----|
| **α\*** | 5.559 | 5.010 | 4.420 | 3.519 | 2.410 |

**핵심: 불균형↑ → 최적 α↓ (덜 공격적).** 극심 IR에서 공격적 가중(높은 α)이 다수 클래스를 과억제 → 최적이 후퇴. network/tabular의 "극단 IR서 wce·pwce 붕괴"와 동일 메커니즘.

**Fit (2026-06):**
- raw-IR 선형: `α* = −0.01572·IR + 5.378`, R²=0.9582
- **log-IR 선형: `α* = −1.017·ln(IR) + 8.071`, R²=0.9633** ← 채택

> ⚠️ **log-IR 채택 근거는 R²가 아님**: ΔR²=0.0052, n=5 → 두 fit은 통계적으로 동등(구분 불가). 채택 사유는 **PLWCE가 log(n) 가중을 쓰므로 α\*를 log(IR)에 선형 표현하는 게 차원적으로 일관**하기 때문. 논문엔 "두 형식 동등하게 적합, log-IR를 보고" 로 정직하게.
>
> ⚠️ **외삽 금지 — CIFAR 범위(IR 10–200) 한정**: log-IR 법칙은 α\*=1(=LWCE) at **IR≈1047**, α\*=0 at **IR≈2798**, IR>2800에선 **음수**(소수클래스 down-weight = 무의미). network/tabular 극단 IR(1366~17,500)엔 적용 불가.

**관측적 정합 (정량 법칙 아님)**: 법칙이 IR>~1047에서 α\*<1을 예측 = "극단 IR에선 더 부드러운 가중이 최적"이라는 **방향**은, network 실측(극단 IR서 wce 꼴찌 붕괴)과 **질적으로 일치**. 단 이는 *"aggressive weighting은 mild~moderate 불균형 전용"* 이라는 **정성적 결론**까지만 — α\*(IR) **정량 법칙 자체는 image LT 밖에서 재현 안 됨**(아래 cross-domain null 참조). 논문에서 "법칙이 도메인 간 일반화"로 과장 금지.

**🔁 cross-domain 재현 (2026-06, 결과: ❌ 법칙 재현 안 됨 — null)**: `alpha_sweep_crossdomain.ipynb` (repo root). 같은 controlled 설계(train만 지수 LT 프로파일로 IR∈{10,20,50,100,200} 통제, test 자연분포 고정)를 tabular credit_card_fraud(binary, N_HEAD=3440)·network CICIDS2018(≥1000 필터→10-class, N_HEAD=8000)에서 PLWCE α(0.3~6.0, 16점) 스윕. **두 도메인 모두 α\*(IR) 법칙 비재현:**
> - **ccf (binary, 깨끗한 null)**: α\* = **0.3(grid 최저)로 전 IR 고정** → IR 무관 + CIFAR와 반대(저IR도 최소 α 선호). fraud는 공격적 reweighting 자체를 안 원함(α→0=CE 쪽). 필터 없이 tail 17개까지 가도 동일 → 신뢰 높음.
> - **CICIDS2018**: α\* = [0.3,6.0,4.1,2.2,4.1] **무작위**(R²=0.02~0.07), **F1이 α 전체에서 0.859~0.871(~1%)로 평탄** → α 레버리지 거의 없음. ⚠️ 단, IR 통제용 ≥1000 필터가 hard tail(α가 먹히는 곳)을 제거한 교란 가능 → ccf null이 더 깨끗한 증거.
>
> **결론: α\*(IR) 법칙은 image LT 특이적, universal 아님.** 의미 = **reweighting 레버리지가 도메인 의존** — image LT는 강(α 민감), IDS/fraud는 약(클래스 분리 쉬움·F1 포화 → "less-is-more"). 이는 network/tabular에서 **wce 최악·제안 손실 gap 작음**과 정합(α-sweep이 인과적으로 확증). **논문**: 위 α\*(IR) 법칙을 universal로 주장 금지 — CIFAR 분석으로 한정하고 *"tabular/IDS에선 F1이 α에 둔감해 재현 안 됨(실측)"* 명시가 정직·방어적. 결과: `network_data/results/mlp/alpha_sweep_{domain}.json`.

---

## 참고 링크

- medical_data/CLAUDE.md — 의료 세그멘테이션 실험 문서
- medical_data/rules.md — 손실함수 설계 및 SoTA 비교
- https://github.com/kaidic/LDAM-DRW — Long-Tail Learning 벤치마크
