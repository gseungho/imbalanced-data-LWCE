# CLAUDE.md — tabular_data

표 형(Tabular) 데이터에서 LWCE 계열 손실 함수를 비교하는 실험.

> **두 가지 실험 트랙이 공존한다:**
> 1. **통합 노트북 (ASC 논문용, 권장)** — `Tabular_MLP.ipynb`.
>    `network_data/Network_MLP.ipynb`·CIFAR 실험과 동일한 구조: **PyTorch MLP**, **8 losses**(ce/wce/pwce/sqce/lwce/plwce/cb/focal, **WCE 포함**),
>    **통일된 Optuna 범위**(PWCE 0.3–5.0, PLWCE 0.5–6.0, **Focal 1.0–5.0**; sqce=√-CE는 α=0.5 고정·Optuna 없음), 5 seeds.
>    한 노트북에서 11개 데이터셋을 `load → Optuna → 학습 → 저장` 루프로 처리. 결과: `results/mlp/`.
> 2. **레거시 XGBoost 트랙 (KISS 추계학술대회용)** — `Imbalanced_Data_Loss.ipynb` + `scr/` 전체.
>    **XGBoost gblinear** + numpy 커스텀 grad/hess 손실. WCE 포함, alpha 단독/2D Optuna.
>    결과: `results/xgboost_gblinear_*.csv`. 아래 "scr/ 모듈"은 이 트랙용.

---

## 통합 노트북 (`Tabular_MLP.ipynb`) — ASC 논문용

### 모델 및 공통 설정
- **모델**: MLP (256→128→64, BatchNorm, ReLU, Dropout=0.3) — network_data와 동일
- **손실 함수 8종**: `ce`, `wce`, `pwce`, `sqce`, `lwce`, `plwce`, `cb`, `focal` (WCE 포함)
  - **논문 역할**: `lwce`/`plwce` = **proposed**, `sqce` = reported baseline(√-CE, w ∝ 1/√n, **α=0.5 고정·Optuna 없음**), `pwce` = **분석/이론 섹션용**(α-sweep foil, main 비교 아님), `ce`/`wce`/`cb`/`focal` = baseline
  - **⚠️ wce 추가 (2026-06, 완료)**: weight-explosion 동기가 정조준하는 표준 베이스라인. `wce = pwce(α=1.0)` 특수해(`total/n_i`), 파라미터·Optuna 없음(Cell 6 resume이 wce×5×11=55 run만 신규 학습). **결과: wce가 F1 최악(7.18/8), 극심 불균형서 붕괴** — 아래 결과 표 참조.
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

### 실험 결과 요약 (11종, 2026-07 재실행, N=5 seeds, F1-Macro)

수치 출처: `results/mlp/mlp_all_results.json` (**11 losses**, gradient/per-class 포함). 605 run 전부 완성.
**proposed = lwce/plwce/eslwce**, combined는 §4.5 ablation, logitadj는 §4.2 baseline.

**🔑 lwce = F1 평균순위 1위 (3.09/11)**, wce = 최하(10.00/11).
평균순위: **lwce 3.09** / eslwce 4.18 / logitadj 4.82 / pwce 4.91 / sqce 5.55 / combined 6.27 /
plwce 6.36 / ce 6.45 / focal 6.64 / cb 7.73 / **wce 10.00**.
평균 F1: **lwce 0.730** > eslwce 0.725 > pwce 0.725 > sqce 0.723 > combined 0.721 > logitadj 0.720 >
plwce 0.719 > focal 0.715 > ce 0.714 > cb 0.701 > **wce 0.683**.
G-Mean은 wce 0.695·cb 0.688·pwce 0.689이 최고지만 F1은 하위 — **wce/cb는 head 희생으로 산 소수 클래스 이득**.

| 데이터셋 | 클래스 | IR | 🥇 최우수 | lwce | plwce | eslwce | combined |
|---------|-------|-----|----------|------|-------|--------|----------|
| credit_card_fraud | 2 | 578 | ce 0.9096 | 0.9086(3) | 0.9081(4) | 0.9094(2) | 0.9080(5) |
| aps_failure | 2 | 59 | pwce 0.9105 | 0.9083(2) | 0.9060(6) | 0.9073(4) | 0.9030(9) |
| bank_marketing | 2 | 7.5 | logitadj 0.7771 | 0.7768(2) | 0.7762(4) | 0.7754(6) | 0.7761(5) |
| telco_churn | 2 | 2.8 | focal 0.7266 | 0.7252(2) | 0.7158(9) | 0.7250(3) | 0.7177(7) |
| german_credit | 2 | 2.3 | **lwce 0.6994** 🥇 | **0.6994(1)** | 0.6869(8) | 0.6979(3) | 0.6881(6) |
| secom | 2 | 14 | cb 0.5804 | 0.5693(6) | 0.5536(9) | 0.5667(7) | 0.5585(8) |
| credit_card_default | 2 | 3.5 | pwce 0.7029 | 0.6828(8) | 0.7001(4) | 0.6833(7) | 0.6998(5) |
| glass | 6 | 8.8 | pwce 0.6794 | 0.6775(2) | 0.6227(5) | 0.6101(9) | 0.6115(8) |
| steel_faults | 7 | 12 | **lwce 0.7684** 🥇 | **0.7684(1)** | 0.7556(7) | 0.7657(3) | 0.7604(5) |
| yeast | 10 | 108 | **eslwce 0.5187** 🥇 | 0.5031(5) | 0.5006(7) | **0.5187(1)** | 0.5014(6) |
| page_blocks | 5 | 172 | **eslwce 0.8177** 🥇 | 0.8142(2) | 0.7860(7) | **0.8177(1)** | 0.8020(5) |

**제안 손실 소견 (v1 8종 대비 변화 포함)**
- **lwce 강세 지속**: F1 평균순위 1위. **단독 1위 2개**(german_credit, steel_faults) + 다수 2~3위. 다중 클래스(glass/steel_faults/page_blocks)에서 최상위권.
- **eslwce가 신규 강자**: 평균순위 2위, **yeast·page_blocks 단독 1위**(둘 다 중~고 IR 다클래스). ES-LWCE가 예상보다 F1에서 선전.
- **plwce는 tabular에서 중위권**(평균순위 6.36) — v1에서 극심 불균형 ccf 1위였으나, 재실행에선 ccf도 ce가 1위(0.9096). **tabular은 이진·저IR이 많아 aggressive 가중의 이점이 작다.**
- **combined ≈ plwce 경향**: 대부분 노이즈 내 차이. Optuna가 combined ε를 낮게 골라 PLWCE로 수렴.
- **저불균형 이진(german_credit/telco_churn/credit_card_default)**: 전 손실 노이즈 수준 → 이점은 **불균형·클래스 수가 클수록** 뚜렷.

> **역할 분담 (재실행 반영)**: lwce·eslwce = **다클래스·중간 불균형 F1** 최상. plwce = tabular에선 상대적 약세이나
> text(다클래스 트랜스포머)·CIFAR-100에선 강세 → **모달리티 의존**. cf. `network_data/CLAUDE.md`, `text_classification/CLAUDE.md`.

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
