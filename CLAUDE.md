# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
│   ├── {Domain}.ipynb            # 도메인 실험 노트북 (8종)
│   ├── results/{도메인}/         # JSON/xlsx/PNG 결과 저장
│   └── boundary_ablation/
│       ├── rules.md              # Boundary ablation 실험 배경 및 설계
│       ├── {Domain}_boundary_ablation.ipynb   # ablation 노트북 (3종: WMH/ISIC/TN3K)
│       └── results/{도메인}/     # ablation 결과
└── tabular_data/
    └── scr/custom_losses.py      # 표 형 데이터용 손실 함수 (별도 버전)
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
- **Boundary Loss 실험** (`boundary_ablation/`): CE를 LWCE/PLWCE로 교체 + Dice + BL/LBL 조합 — Dice와 중복 없는 방식으로 LWCE 적용
  - `ce_dice_boundary` = CE + Dice + BL (문헌 베이스라인, Kervadec 2019)
  - `plwce_dice_boundary` = PLWCE + Dice + BL (핵심 실험)
  - `plwce_dice_log_boundary` = PLWCE + Dice + LBL (log scaling 확장)
  - `plwce_boundary` / `plwce_log_boundary` = Dice 제거 실험 (성능 하락 예상)
  - **LBL (Log-Boundary Loss)**: 거리 맵에 `log(1+|d|)` 적용 — LWCE의 log scaling 철학을 픽셀 거리 도메인에 확장
  - **BL/LBL Loss 음수 정상**: 모델이 병변 내부(d<0)를 잘 예측할수록 loss가 음수로 수렴 — 정상 동작

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
- proxy 설정: `subset_ratio=0.15`, `epochs=5`, `n_trials=20`

### 데이터 로딩
- WMH: `kagglehub.dataset_download("farahmo/wmh-dataset")` 자동 다운로드
- Pancreas: Google Drive `synapse/synapse.zip` → `/tmp/` 압축 해제
- 기타: Google Drive `MyDrive/imbalanced-data-LWCE/{domain}/` → `/tmp/` 복사
- WMH 입력: 3채널 `[FLAIR, T1, FLAIR]` (ImageNet 인코더 3ch 맞춤)
- WMH BG-only 슬라이스: `BG_ONLY_RATIO=0.3` (WMH 슬라이스 수의 30% 포함)

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
