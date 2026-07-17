# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# ⚠️ ASC 논문 실험 (2026-07 재구성) — 먼저 읽을 것

논문에 들어가는 도메인은 **4개뿐**: `image_classification`, `tabular_data`, `network_data`, `text_classification`.
(`medical_data`·`image_detection`은 이 논문 범위 밖 — 아래 규칙 적용 안 됨.)

## 공유 모듈은 리포 루트

```
imbalanced-data-LWCE/           ← Colab: /content/drive/MyDrive/imbalanced-data-LWCE
├── custom_losses.py            # 분류용 손실 11종 (4개 도메인 공유)
├── experiment_utils.py         # GradLogger / extended_metrics / OPTUNA_GRIDS (4개 도메인 공유)
├── image_classification/       # resnet32.py (도메인 전용 백본)
├── text_classification/        # transformer_text.py (도메인 전용 백본)
├── network_data/src/           # data_handler.py     ← 'scr' 아님 (2026-07 rename)
└── tabular_data/src/           # data_handler.py + custom_losses.py(XGBoost 레거시)
```

**모든 노트북 Cell 0은 Colab 전용**(로컬 분기 없음):
```python
from google.colab import drive; drive.mount('/content/drive')
REPO = '/content/drive/MyDrive/imbalanced-data-LWCE'
sys.path.insert(0, f'{REPO}/<domain>/src')   # 또는 백본 폴더
sys.path.insert(0, REPO)                     # ← 마지막에 insert해야 루트가 우선
```

> ⚠️ **`tabular_data/src/custom_losses.py`는 XGBoost 레거시판으로 루트와 이름이 충돌한다.**
> `insert(0)`은 나중 호출이 앞에 오므로 **REPO를 반드시 마지막에 insert**할 것.

## 손실 11종 (전 도메인 동일)

`ce, wce, pwce, sqce, lwce, plwce, eslwce, combined, cb, focal, logitadj`
**proposed = lwce / plwce / eslwce**, `combined`는 §4.5 ablation 전용, `logitadj`는 §4.2 baseline.

**파싱 우선순위**: `combined > plwce > eslwce > lwce > pwce > sqce > wce > cb > ce`
> `'eslwce'`가 `'lwce'`를 부분문자열로 포함 → 반드시 `lwce`보다 먼저 검사.

## Optuna (전 도메인 통일)

`experiment_utils.OPTUNA_GRIDS` 하나로 관리. **1D 30 trials, `combined`만 2D 10×6=60**.
→ 데이터셋당 proxy 학습 210회. `n_trials`는 grid 크기와 **정확히** 일치해야 함(`grid_n_trials()`).
목적함수는 **F1-Macro** (balanced_acc 아님 — 옛 문서의 오기 주의).

## 3차 피드백 §4.7 gradient 계측 — 사후 복구 불가

`GradLogger`로 epoch별 `grad_norm` / `grad_ratio(Few/Many)` 기록. **반드시 학습 루프 안**:
```python
logits = model(xb); logits.retain_grad()      # ← 없으면 GradLogger가 즉시 RuntimeError
loss.backward(); glog.update(logits, yb, model); optimizer.step()
```
나머지(G-Mean·Worst-class·Minority Recall·Weight Distribution·통계검정)는 **전부 사후 계산 가능**.

## Weight Normalization — 이미 충족

3차 §3.3이 명시를 요구한 `w̃_c = C·w_c/Σⱼwⱼ`가 `calculate_weights()`의 `weights/np.mean(weights)`와
**수학적으로 동일**하고 **모든 모드에 무조건 적용**됨 → WCE·CB도 동일 scale convention(공정 비교 충족).
**논문에 문장만 추가하면 됨. 코드 수정 불필요.**

## ⚠️ 결과 파일

**버전 접미사 없이 기존 위치에 저장**. 11종 + gradient/per-class/history라 **옛 8종 파일과 스키마가 다름**
→ Drive의 기존 결과를 정리(보관)한 뒤 실행할 것. **각 도메인 CLAUDE.md의 결과표는 옛 8종 수치이므로
재실행 후 전부 갱신 필요.**

---

# 프로젝트 컨텍스트

불균형 클래스 환경에서 의료 이미지 세그멘테이션 성능을 검증하는 연구 프로젝트.
**커스텀 손실 함수(LWCE 계열)**를 다양한 의료 영상 도메인에 적용·비교함.

실험 결과 상세 요약은 `medical_data/results/experiment_summary.md` 참조.
도메인별 상세 설정·SoTA 비교는 `medical_data/rules.md` 참조.
Boundary ablation 실험 배경/설계는 `medical_data/boundary_ablation/rules.md` 참조.

---

## 아키텍처 개요

```
imbalanced-data-LWCE/
├── medical_data/
│   ├── custom_losses.py          # 핵심 모듈 — 모든 손실 함수 정의
│   ├── rules.md                  # 도메인별 실험 설계 및 SoTA 비교
│   ├── {Domain}.ipynb            # 도메인 실험 노트북 (11종)
│   ├── results/{도메인}/         # JSON/xlsx/PNG 결과 저장
│   │   └── experiment_summary.md # 전체 실험 결과 요약
│   └── boundary_ablation/
│       ├── rules.md              # Boundary ablation 실험 배경 및 설계
│       ├── {Domain}_boundary_ablation.ipynb   # ablation 노트북 (3종: WMH/ISIC/TN3K)
│       └── results/{도메인}/     # ablation 결과
├── tabular_data/
│   ├── Imbalanced_Data_Loss.ipynb  # 표 형 데이터 실험 노트북
│   └── scr/
│       ├── custom_losses.py      # XGBoost용 커스텀 grad/hess 손실 (numpy 기반)
│       ├── data_handler.py       # 데이터 로드 및 전처리
│       ├── evaluation.py         # 평가 지표
│       ├── optuna_tuner_alpha_only.py   # alpha 단독 Optuna 탐색
│       └── optuna_tuner_xgb_linear.py  # XGBoost linear booster Optuna 탐색
└── papers/                       # 참고 문헌 PDF
```

### custom_losses.py 구조 (`medical_data/`)

```
calculate_weights()           # 가중치 모드별 계산 (ce/wce/pwce/lwce/plwce/cb)
compute_signed_distance_map() # Boundary Loss용 부호 있는 거리 맵
DiceLoss                      # region-based (background 제외)
TverskyLoss                   # FN 패널티 강화 (α=0.7, β=0.3)
FocalTverskyLoss              # (1 - Tversky)^γ
BoundaryLoss                  # Kervadec 2019, fg_prob × signed_dist_map
LogBoundaryLoss               # log(1+|d|) 거리 압축 버전 (신규 제안)
FocalLoss                     # CE 기반, (1-pt)^γ × CE
SegmentationLoss              # 마스터 클래스 — loss_name 문자열 파싱으로 조합
get_loss_function()           # Factory 함수 (항상 이걸 통해 생성)
```

**loss_name 파싱 규칙**: `{weight_mode}_{region_loss}_{boundary_loss}`
- weight 우선순위: `plwce > lwce > pwce > wce > cb > ce`
- `log_boundary`를 `boundary` 보다 먼저 확인 (substring 충돌 방지)
- λ는 활성 컴포넌트 수 n으로 균등 분배 (1/n), Boundary annealing 시 `set_boundary_alpha(α_t)` 호출

### 도메인 노트북 셀 구성 (표준)

| 셀 | 내용 |
|----|------|
| Cell 0 | 환경설정 (pip install, import, device, RESULTS_DIR) |
| Cell 1 | 데이터 로드 및 DataLoader |
| Cell 2 | 클래스 비율 계산 (`class_counts`) |
| Cell 3 | 모델 정의 (`build_model()`, `to_2ch_logits()`, `compute_val_*()`) |
| Cell 4 | `train_model()` 함수 정의 |
| Cell 5 | Optuna alpha/gamma 탐색 |
| Cell 6 | 전체 실험 실행 |
| Cell 7 | 평가, 시각화, 결과 저장 |

---

## 핵심 규칙

### 손실 함수
- 손실 함수는 항상 `get_loss_function(loss_name, class_counts, alpha, gamma)` 를 통해 생성
- `to_2ch_logits(p) = torch.cat([-p, p], dim=1)` 패턴 사용 — `[zeros, p]` 방식 금지
- Optuna로 alpha/gamma 탐색 후 최종 실험 진행
- **세그멘테이션 인사이트**: Dice Loss는 분모 정규화로 소수 클래스 무시를 이미 방지 → LWCE+Dice 조합에서 LWCE 효과가 희석됨 (실험으로 확인)
- **Boundary Loss 실험** (`boundary_ablation/`): CE를 LWCE/PLWCE로 교체 + Dice + BL 조합 — Dice와 중복 없는 방식으로 LWCE 적용
  - `ce_dice` = CE + Dice (기존 베이스라인)
  - `plwce_dice` = PLWCE + Dice (기존 PLWCE 베스트)
  - `ce_dice_boundary` = CE + Dice + BL (문헌 베이스라인, Kervadec 2019)
  - `plwce_dice_boundary` = PLWCE + Dice + BL (핵심 실험)
  - `plwce_boundary` = PLWCE + BL (Dice 제거 실험, 성능 하락 예상)
  - **LBL 제거**: LBL(Log-Boundary Loss)은 BL 대비 유의미한 성능 차이 없어 실험군에서 제외
  - **BL Loss 음수 정상**: 모델이 병변 내부(d<0)를 잘 예측할수록 loss가 음수로 수렴 — 정상 동작

### 경로 규칙
- custom_losses import: `sys.path.insert(0, '/root/imbalanced-data-LWCE/medical_data')`
- 오타 주의: `imbalanced` (올바름) vs `inbalanced` (오타, 기존 일부 파일에 존재)
- 결과 저장: `medical_data/results/{도메인}/`
- Boundary ablation 결과: `medical_data/boundary_ablation/results/{도메인}/`
- 임시 파일(체크포인트, 캐시): `/tmp/`

### Optuna
- `TQDM_DISABLE` 환경변수로 trial 중 tqdm 출력 억제: `os.environ['TQDM_DISABLE'] = '1'`
- `if t.value is not None` — `if t.value` 사용 금지 (Dice=0.0 trial 필터링 버그)
- `study_pf` 시각화 코드는 반드시 `study_pf = optuna.create_study(...)` 정의 이후에 위치
- alpha 범위: 2.5~15.0 (극심한 불균형 도메인은 ~20.0까지 확장)
- proxy 설정: `subset_ratio=0.15`, `epochs=8`, `n_trials=20`
- **GridSampler 사용**: TPE 대신 균일 분할 `GridSampler`로 전체 범위 탐색
  - 1D (alpha only): `GridSampler({'alpha': np.linspace(LOW, HIGH, N_TRIALS).tolist()})`
  - 2D (alpha+gamma): `GridSampler({'alpha': ..., 'gamma': ...})` — N_TRIALS_PF=40→8×5, 60→10×6
  - `n_trials` 값이 grid 크기와 정확히 일치해야 함 (N_TRIALS_PF는 명시적 int로 지정)
  - GridSampler와 MedianPruner 병행 금지 (Kvasir/Retinal 등에서 pruner 제거됨)

### 데이터 로딩
- WMH: `kagglehub.dataset_download("farahmo/wmh-dataset")` 자동 다운로드
- Pancreas: Google Drive `synapse/synapse.zip` → `/tmp/` 압축 해제
- 기타: Google Drive `MyDrive/imbalanced-data-LWCE/{domain}/` → `/tmp/` 복사
- WMH 입력: 3채널 `[FLAIR, T1, FLAIR]` (ImageNet 인코더 3ch 맞춤)
- WMH BG-only 슬라이스: `BG_ONLY_RATIO=0.3` (WMH 슬라이스 수의 30% 포함)

### tabular_data 손실 함수 (`tabular_data/scr/custom_losses.py`)
- PyTorch 대신 **numpy 기반**, XGBoost 커스텀 objective용 grad/hess 반환
- 클래스: `WeightedCrossEntropy`, `PowerScaledInverseCE`, `LogWeightedCE`, `PowerLogWeightedCE`
- `initialize(y_true)` 호출로 가중치 초기화 후 `compute_grad_hess(y_true, y_pred)` 사용
- medical_data의 `get_loss_function()` 팩토리 패턴과 별개 — 두 버전 혼용 금지

---

## 알려진 버그 패턴 (반복 주의)

| 패턴 | 잘못된 방식 | 올바른 방식 |
|------|------------|------------|
| Optuna trial 필터 | `if t.value:` | `if t.value is not None:` |
| study 시각화 순서 | study 정의 전에 plot 코드 | study 정의 후에 plot 코드 |
| 1채널 logit 변환 | `torch.cat([zeros, p])` | `torch.cat([-p, p])` |
| albumentations crop | `height=H, width=W` | `size=(H, W)` |
| TransUNet import | `sys.path.add(networks/)` + `from . import` | `sys.path.add(TransUNet/)` + `from networks.xxx import` |
| 학습 전 모델 호출 | Cell 정의 마지막에 `build_model()` 호출 | 정의만 하고 호출 금지 (불필요한 weight 다운로드) |
| DataLoader num_workers | `NUM_WORKERS = 2` (또는 그 이상) | `NUM_WORKERS = 0` — Colab notebook에서 반복 학습 시 worker 프로세스 정리 오류 발생 |
| 평가 지표명 | `'Sens'`, `'Spec'` 키 사용 | `'Sensitivity'`, `'Specificity'` — JSON/Excel 키와 통일, 전 노트북 표준 |

---

## 실험 현황 요약 (2026-03-22 기준)

| 도메인 | 불균형 | 🥇 최우수 | CE 대비 Dice |
|--------|--------|----------|------------|
| MoNuSeg (Nuclei) | 2.6:1 | `ce_dice` | ±0.000 |
| ISIC 2018 (Skin) | 3.6:1 | `lwce_dice` | +0.006 |
| Kvasir-SEG (Polyp) | ~5:1 | `lwce_dice` | +0.005 |
| TN3K (Thyroid) | 6:1 | `ce_dice` | ±0.000 |
| DRIVE (Retinal) | ~9:1 | `plwce_focal_dice` | +0.005 |
| LiTS (Liver/Tumor) | 15~356:1 | `ce_dice`/`plwce_dice` | Tumor +0.013 |
| Pancreas (8-class) | 다중 | `plwce_dice` | +0.003 mDice |
| WMH (Brain) | 250:1 | `plwce_dice` | +0.010 |
| BUSI (Breast US) | ~4:1 | — | 진행 중 |
| ACDC (Cardiac MRI) | ~9:1 | — | 진행 중 |
| REFUGE (Optic) | Disc ~5:1 | — | 진행 중 |

**소견**: 불균형 ≤5:1 → 효과 미미. 불균형 >50:1 → `plwce_dice` 일관적으로 우수. `pwce_dice`는 Pancreas에서 담낭 Dice 0.001 붕괴 사례 있음 — 주의.
