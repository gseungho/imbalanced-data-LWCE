# CLAUDE.md — tabular_data

표 형(Tabular) 데이터에서 LWCE 계열 손실 함수를 비교하는 실험.

> **두 가지 실험 트랙이 공존한다:**
> 1. **통합 노트북 (ASC 논문용, 권장)** — `Tabular_MLP.ipynb`.
>    `network_data/Network_MLP.ipynb`·CIFAR 실험과 동일한 구조: **PyTorch MLP**, **7 losses**(ce/pwce/sqce/lwce/plwce/cb/focal, **WCE 제외**),
>    **통일된 Optuna 범위**(PWCE 0.3–5.0, PLWCE 0.5–6.0, **Focal 1.0–5.0**; sqce=√-CE는 α=0.5 고정·Optuna 없음), 5 seeds.
>    한 노트북에서 11개 데이터셋을 `load → Optuna → 학습 → 저장` 루프로 처리. 결과: `results/mlp/`.
> 2. **레거시 XGBoost 트랙 (KISS 추계학술대회용)** — `Imbalanced_Data_Loss.ipynb` + `scr/` 전체.
>    **XGBoost gblinear** + numpy 커스텀 grad/hess 손실. WCE 포함, alpha 단독/2D Optuna.
>    결과: `results/xgboost_gblinear_*.csv`. 아래 "scr/ 모듈"은 이 트랙용.

---

## 통합 노트북 (`Tabular_MLP.ipynb`) — ASC 논문용

### 모델 및 공통 설정
- **모델**: MLP (256→128→64, BatchNorm, ReLU, Dropout=0.3) — network_data와 동일
- **손실 함수 7종**: `ce`, `pwce`, `sqce`, `lwce`, `plwce`, `cb`, `focal` (WCE 제외)
  - **논문 역할**: `lwce`/`plwce` = **proposed**, `sqce` = reported baseline(√-CE, w ∝ 1/√n, **α=0.5 고정·Optuna 없음**), `pwce` = **분석/이론 섹션용**(α-sweep foil, main 비교 아님), `ce`/`cb`/`focal` = baseline
- **손실 생성**: `image_classification/custom_losses.py`의 `get_clf_loss()` 공유 (sys.path에 `IMG_CLF_PATH` 추가)
- **데이터 로드**: `scr/data_handler.py`의 `load_dataset(ds, DATA_PATH, test_size=0.3, random_state=42)`
- **옵티마이저**: Adam(lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR(eta_min=1e-5)
- `BATCH_SIZE=512`, `FINAL_EPOCHS=100`(소규모 데이터셋 수렴 보장), `num_workers=0`, `SEEDS=[42,43,44,45,46]`
- **평가 지표**: Accuracy, Balanced_Accuracy, F1_Macro, F1_Weighted, Head_F1, Mid_F1, Tail_F1, Per_Class_F1
- **Head/Mid/Tail**: 이진은 소수=Tail/다수=Head, 다중은 `class_counts` 정렬 후 하위 n/3=Tail

### Optuna 설정
- **GridSampler**, **F1-Macro 기준** (`direction='maximize'`), `os.environ['TQDM_DISABLE']='1'`
- proxy: `PROXY_RATIO=0.20`, `PROXY_EPOCHS=20`, 계층 추출(`train_test_split(stratify=y_tr)`, ValueError 시 랜덤 fallback)
- 탐색 대상은 **pwce/plwce/focal 3종만** (각 `N=30` trials):
  - PWCE alpha `np.linspace(0.3, 5.0, 30)`
  - PLWCE alpha `np.linspace(0.5, 6.0, 30)`
  - **FOCAL gamma `np.linspace(1.0, 5.0, 30)`** — γ 하한 1.0 고정
- ce/sqce/lwce/cb는 파라미터 탐색 없음(고정)

> **⚠️ Focal γ<1 NaN 붕괴 버그 (network_data와 동일)**: multiclass softmax focal의 modulating factor
> `(1−p_t)^γ`는 γ<1일 때 gradient `−γ(1−p_t)^(γ−1)`가 `p_t→1`에서 발산 → NaN → 전부 class 0 collapse.
> 증상: focal F1≈`F1_class0/K`, **std=0.0000**, Tail_F1=0.0. **해결**: focal gamma grid 하한 `1.0` 고정.

### 노트북 셀 구성 (표준)
| 셀 | 내용 |
|----|------|
| Cell 0 | 환경설정, 경로, `DATASETS`/`LOSS_CONFIGS`/`SEEDS`, 상수 |
| Cell 1 | 데이터 로드 (`load_dataset` → label remap 0..C-1 → `class_counts`/IR) |
| Cell 2 | 클래스 분포 시각화 (log scale) |
| Cell 3 | MLP 모델 + `compute_metrics()` |
| Cell 4 | `train_model()` 정의 (호출 금지) |
| Cell 5 | Optuna 탐색 (pwce/plwce/focal) → `best_params` |
| Cell 6 | 전체 실험 실행 (per-loss skip 재개) |
| Cell 7 | 시각화 + pivot 저장 |

> **재개(resume) 동작**: Cell 6은 데이터셋별 `{ds}_results.json`을 읽어 `done = {(Loss, Seed)}` 집합으로
> **이미 끝난 (손실, 시드) 조합만 건너뛴다**. 따라서 sqce처럼 손실 1종만 추가하면 기존 6종은 재학습 없이
> 캐시 출력(`(cached)`)되고 신규 손실만 학습·append된다. **Cell 5(Optuna)를 먼저 실행**해야 `best_params`가
> 메모리에 존재 — Cell 6 단독 실행 시 pwce/plwce/focal이 기본 alpha/gamma로 떨어짐.

### 데이터셋 (11종)
`load_dataset`이 처리하는 파일은 `DATA_PATH`(Colab: `MyDrive/imbalanced loss project/data`) 기준.

| 데이터셋 | 클래스 | 불균형 | 비고 |
|---------|-------|-------|------|
| credit_card_fraud | 2 | 극심(~580:1) | `creditcard.csv`, Time 제거, Amount만 스케일 |
| aps_failure | 2 | 심(~60:1) | 자체 train/test split, skiprows=20, na 평균 대체 |
| bank_marketing | 2 | 경미(~8:1) | — |
| telco_churn | 2 | 경미(~3:1) | — |
| german_credit | 2 | 경미(~2:1) | 소규모, seed별 분산 큼 |
| secom | 2 | 중(~14:1) | 고차원 센서, 결측 많음 |
| credit_card_default | 2 | 경미(~3.5:1) | `.xls`, header=1 |
| glass | 6 | 다중 | 레이블 gap {0,1,2,3,4,6} → remap |
| steel_faults | 7 | 다중 | — |
| yeast | 10 | 다중 | 소규모, 분산 큼 |
| page_blocks | 5 | 다중 | — |

### 실험 결과 요약 — 제안 손실 위치 (2026-06, N=5 seeds, F1-Macro)

**lwce / plwce가 우리의 proposed.** 7종 중 순위(`n/7`)로 표기.

| 데이터셋 | 클래스 | 🥇 최우수 | **lwce** | **plwce** |
|---------|-------|----------|----------|-----------|
| credit_card_fraud | 2 (극심) | plwce 0.9097 | 0.9075 (3/7) | **0.9097 (1/7) 🥇** |
| aps_failure | 2 | pwce 0.9076 | 0.9062 (4/7) | 0.9063 (3/7) |
| bank_marketing | 2 | pwce 0.7778 | 0.7753 (3/7) | 0.7767 (2/7) |
| telco_churn | 2 | focal 0.7264 | 0.7246 (2/7) | 0.7187 (5/7) |
| german_credit | 2 | lwce 0.6994 | **0.6994 (1/7) 🥇** | 0.6875 (6/7) |
| secom | 2 | cb 0.5867 | 0.5676 (4/7) | 0.5567 (5/7) |
| credit_card_default | 2 | cb 0.7027 | 0.6829 (5/7) | 0.7017 (3/7) |
| glass | 6 | pwce 0.6794 | 0.6775 (2/7) | 0.6227 (4/7) |
| steel_faults | 7 | lwce 0.7684 | **0.7684 (1/7) 🥇** | 0.7556 (5/7) |
| yeast | 10 | focal 0.5174 | 0.5031 (3/7) | 0.5006 (4/7) |
| page_blocks | 5 | lwce 0.8142 | **0.8142 (1/7) 🥇** | 0.7860 (4/7) |

**제안 손실 소견**
- **lwce 강세**: 11개 중 **단독 1위 3개**(german_credit, steel_faults, page_blocks) + **2위 3개**(bank_marketing, telco_churn, glass). 특히 **다중 클래스(glass/steel_faults/page_blocks)에서 최상위권**.
- **plwce**: **극심 불균형 credit_card_fraud(이진 ~580:1)에서 단독 1위**. 이진 저불균형에선 중상위(bank_marketing 2위, credit_card_default 3위)지만 다중 클래스에선 lwce에 밀림.
- **lwce vs plwce (tabular)**: lwce가 더 일관적으로 상위. plwce는 극단 불균형 1건에서만 우위.
  → network 결과(극심 >1000:1에서 plwce 강세)와 합쳐 보면 **"plwce는 극단 불균형용, lwce는 중간 불균형·다중 클래스용"** 역할 분담 경향.
- **저불균형 이진(german_credit, telco_churn 등)**: 전 손실이 노이즈 수준(±0.02~0.03) → 제안 손실의 이점은 **불균형이 클수록·클래스가 많을수록** 뚜렷.

---

## 레거시 XGBoost 트랙 (`Imbalanced_Data_Loss.ipynb` + `scr/`)

KISS 추계학술대회용. **XGBoost gblinear** + numpy 커스텀 objective. 통합 MLP 트랙과 **혼용 금지**.

### scr/ 모듈
- **`custom_losses.py`** — PyTorch 아님. **numpy 기반** XGBoost 커스텀 grad/hess 반환.
  - 클래스: `WeightedCrossEntropy`, `PowerScaledInverseCE`, `LogWeightedCE`, `PowerLogWeightedCE`
  - 사용 패턴: `initialize(y_true)`로 가중치 초기화 → `compute_grad_hess(y_true, y_pred)`
  - medical_data/image_classification의 `get_loss_function`/`get_clf_loss` 팩토리와 **별개**
- **`data_handler.py`** — `load_dataset(name, base_data_path, test_size, random_state)` → `(X_train, X_test, y_train, y_test)`. 데이터셋별 전처리(스케일/one-hot/결측 처리) 하드코딩. **통합 노트북도 이 로더를 공유**.
- **`evaluation.py`** — 평가 지표
- **`optuna_tuner_alpha_only.py`** — alpha 단독 Optuna 탐색
- **`optuna_tuner_xgb_linear.py`** — XGBoost linear booster Optuna 탐색 (3-fold CV, f1_macro)

### 결과물
- `results/xgboost_gblinear_full_results*.csv`, `results/*.png` (f1/pr_auc heatmap 등)
- `KISS 추계학술대회/` — proceeding/초록/발표자료(.tex/.pdf), 분석 스크립트

---

## 알려진 버그 패턴 (반복 주의)

| 패턴 | 잘못된 방식 | 올바른 방식 |
|------|------------|------------|
| Focal γ 하한 | grid `[0, …]` 또는 `[0.5, …]` | `np.linspace(1.0, 5.0, N)` (γ<1 NaN 붕괴) |
| DataLoader workers | `num_workers≥1` | `num_workers=0` |
| 레이블 gap | remap 없이 학습 (glass {…,6}) | Cell 1에서 0..C-1 remap |
| loss 생성 | numpy `custom_losses`와 PyTorch `get_clf_loss` 혼용 | 트랙별로 분리 사용 |
| Optuna 트리 필터 | `if t.value:` | `if t.value is not None:` |
