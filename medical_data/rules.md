# Medical Data 불균형 세그멘테이션 연구 규칙 및 계획서

## 1. 프로젝트 개요

불균형 클래스 환경에서 의료 이미지 세그멘테이션 성능을 향상시키기 위해
**커스텀 손실 함수(LWCE 계열)**의 효과를 다양한 의료 영상 도메인에 걸쳐 검증하는 연구.

---

## 2. 핵심 모듈 구조 (`custom_losses.py`)

### 2-1. 가중치 모드 (`calculate_weights`)

| 모드 | 수식 | 특징 |
|------|------|------|
| `ce` | w = 1 | 기준선 (가중치 없음) |
| `wce` | w = total / count | 단순 역비율 |
| `pwce` | w = (total/count)^α | alpha로 강도 조절 |
| `lwce` | w = 1 / log(1 + count) | 로그 스케일 완화 |
| `plwce` | w = 1 / log(1 + count)^α | alpha로 강도 조절 |
| `plwce_focal` | PLWCE 가중치 + Focal Loss | alpha(가중치 강도) + gamma(focal 강도) 2개 파라미터 |
| `cb` | w = (1-β) / (1-β^n) | Effective Number 기반 |

- 모든 가중치는 **평균으로 정규화** (`weights / mean(weights)`)
- `alpha` 범위: 실험상 2.5~15.0 (Optuna로 탐색; 극심한 불균형 도메인은 ~20.0까지 확장)
- `beta` 기본값: 0.9999 (CB Loss용)
- `gamma` 범위: 0.5~5.0 (Focal Loss용; Optuna로 탐색)

### 2-2. Loss 클래스 구조

```
SegmentationLoss (마스터 클래스)
├── weight_mode 파싱  →  calculate_weights()
├── main_loss: FocalLoss 또는 CrossEntropyLoss (가중치 적용)
└── dice_loss: DiceLoss (background 제외)
```

### 2-3. Loss 명명 규칙 (문자열 파싱)

```
{weight_mode}_{loss_type}
예: lwce_dice, plwce_focal_dice, ce_dice, cb_dice
```

- `dice` 포함 → CE + Dice 혼합 (각 0.5 비율)
- `focal` 포함 → CE 대신 Focal Loss 사용
- 가중치 키워드는 `plwce > lwce > pwce > wce > cb` 순으로 파싱 (우선순위)

#### 도메인별 적용 손실 함수 현황

| 도메인 | Binary Loss 목록 | Optuna 탐색 대상 |
|--------|-----------------|-----------------|
| Skin_Lesion_ISIC2018 | `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`, `cb_dice`, `plwce_focal_dice` | PLWCE α, PWCE α, PLWCE+Focal α+γ |
| TN3K_Thyroid_Ultrasound | `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`, `cb_dice`, `plwce_focal_dice` | PLWCE α, PWCE α, PLWCE+Focal α+γ |
| WMH_Brain_Lesion_MRI | `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`, `cb_dice`, `plwce_focal_dice` | PLWCE α, PWCE α, PLWCE+Focal α+γ |
| MoNuSeg_Nuclei_Pathology | `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`, `cb_dice`, `plwce_focal_dice` | PLWCE α, PWCE α, PLWCE+Focal α+γ |
| Pancreas_MultiOrgan_CT | `ce_dice`, `plwce_dice`, `pwce_dice`, `plwce_focal_dice` | PLWCE α, PWCE α, PLWCE+Focal α+γ |

> **원칙**: `plwce_focal_dice`는 모든 도메인에 추가. PWCE α 탐색도 함께 진행.

### 2-4. Factory 함수

```python
criterion = get_loss_function(loss_name, class_counts, alpha=1.0, beta=0.9999, gamma=2.0)
```

**항상 이 함수를 통해 손실 함수를 생성할 것.**

---

## 3. 완료된 실험 도메인

### 3-1. Endoscopic Polyp Image (Kvasir-SEG)
- **모델**: PraNet (Res2Net50 백본)
- **태스크**: Binary segmentation (폴립 vs 배경)
- **데이터**: 1,000장 (Train 800 / Val 200), 입력 352×352
- **불균형**: BG:FG = **5.4:1** (비교적 완만)
- **실험 Loss**: `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`, `cb_dice`
- **핵심 트릭**: `to_2ch_logits(p) = torch.cat([-p, p], dim=1)` — 1채널 sigmoid 출력을 2채널 logit으로 변환
- **SoTA 참고** (2026년 3월 인터넷 조사 기준):
  | 방법 | Dice | IoU | 출처 |
  |------|------|-----|------|
  | MNet-SAt (2024) | **96.61%** | — | IEEE Access |
  | ARCUNet (2025) | 95.34% | **93.53%** | arXiv |
  | PraNet (2021, 원본) | 89.8% | 84.0% | MICCAI'21 |
  | U-Net baseline | ~79~82% | — | 복수 논문 |

### 3-2. Retinal Vessel Image (DRIVE)
- **모델**: IterNet (경량 CNN) — PraNet도 비교 가능
- **태스크**: Binary segmentation (혈관 vs 배경)
- **데이터**: DRIVE 20장 (train/val split), test에 GT 없음
- **불균형**: BG:FG = **10.7:1** (심각)
- **전처리**: CLAHE (Green 채널 강조), 256×256 Patch 기반 샘플링
- **실험 Loss**: `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`
- **결과**: `lwce_dice` 최고 (Dice=0.7033)
- **평가 지표**: Dice, Sensitivity, Specificity, AUC
- **SoTA 참고** (2026년 3월 인터넷 조사 기준):
  | 방법 | Dice | AUC | Sensitivity | Specificity | 출처 |
  |------|------|-----|-------------|-------------|------|
  | TransUNext (2024) | — | **0.9867** | 0.8208 | 0.9840 | Computers in Biology |
  | GAN+UNet (2024) | 0.8215 | 0.9772 | 0.8301 | 0.9781 | IEEE JBHI |
  | LU-RA Transformer (2025) | 0.7871 | — | — | — | Applied Intelligence |
  | IterNet (원본) | ~0.779 | ~0.979 | — | — | AAAI'20 |
  | U-Net baseline | ~0.730 | ~0.974 | — | — | 복수 논문 |

### 3-3. Pancreas / Multi-Organ CT (Synapse)
- **모델**: TransUNet (R50+ViT-B/16)
- **태스크**: 9-class segmentation (배경 + 8개 장기)
- **데이터**: Synapse Multi-organ CT (.npz 슬라이스 / .h5 볼륨)
- **불균형**: 췌장 등 소수 장기 극심한 불균형
- **하이퍼파라미터 탐색**: Optuna (alpha 범위 2.5~15.0, 30 trials / PLWCE+Focal 60 trials, subset_ratio=0.15)
- **실험 Loss**: `ce_dice`, `plwce_dice`, `pwce_dice`
- **평가 지표**: 클래스별 Dice + mDice (background 제외)
- **SoTA 참고** (2026년 3월 인터넷 조사 기준):
  | 방법 | mDice (%) | HD95 (mm) | 출처 |
  |------|-----------|-----------|------|
  | DS-UNETR++ (2025) | **87.75** | **6.67** | arXiv |
  | DIN (2025) | 85.49 | 10.74 | Medical Image Analysis |
  | SwinUNet (2021) | 79.13 | 21.55 | ECCV'22 |
  | TransUNet (2021, baseline) | 77.48 | 31.69 | arXiv |
  | U-Net baseline | ~68~74 | ~39 | 복수 논문 |

---

## 4. 추가 연구 계획 (신규 도메인)

### 4-0. 완료된 신규 도메인 (코드 완성, 실행 대기)

| 도메인 | 파일 | 모달리티 | 모델 | 불균형 | 상태 |
|--------|------|---------|------|--------|------|
| 피부 병변 (ISIC 2018) | `Skin_Lesion_ISIC2018.ipynb` | Dermoscopy(RGB) | U-Net (ResNet34) | 중간 | 코드 완성 |
| 간/종양 CT (LiTS 2017) | `LiTS_Liver_Tumor.ipynb` | CT (3-class) | U-Net++ (ResNet50) | BG:Tumor=356:1 | 코드 완성 |
| 갑상선 결절 초음파 (TN3K) | `TN3K_Thyroid_Ultrasound.ipynb` | Ultrasound | U-Net (ResNet34) | ~20:1 | 코드 완성 |
| 뇌 백질 병변 MRI (WMH 2017) | `WMH_Brain_Lesion_MRI.ipynb` | MRI (FLAIR+T1) | U-Net (ResNet34) | ~100~600:1 | 코드 완성 |
| 조직병리 세포핵 (MoNuSeg 2018) | `MoNuSeg_Nuclei_Pathology.ipynb` | H&E Pathology | U-Net (ResNet34) | ~3~8:1 | 코드 완성 |

**피부 병변 (ISIC 2018) SoTA 참고** (2026년 3월 인터넷 조사 기준):
| 방법 | Dice | IoU | 출처 |
|------|------|-----|------|
| ARCUNet (2025) | **95.34%** | **93.53%** | arXiv |
| Meta-UNet (2025) | 93.14% | 88.21% | Applied Sciences |
| UNet++ (2019) | ~89.0% | ~83.5% | MICCAI'18 |
| U-Net baseline | ~85~88% | — | 복수 논문 |

**간/종양 CT (LiTS 2017) SoTA 참고** (2026년 3월 인터넷 조사 기준):
| 방법 | 간 Dice | 종양 Dice | 출처 |
|------|---------|-----------|------|
| ASLseg (2024) | — | **74.28%** | Medical Image Analysis |
| nnU-Net (2021) | **~98.94%** | ~70~75% | Nature Methods |
| Swin-UNet (2021) | ~95.6% | ~68.1% | ECCV'22 |
| U-Net++ baseline | ~95% | ~60~65% | 복수 논문 |
> ⚠️ LiTS 종양 Dice는 크기·수에 따라 편차 극심 (소형 종양에서 크게 하락)

---

### 4-0-A. 갑상선 결절 초음파 (TN3K)

- **파일**: `TN3K_Thyroid_Ultrasound.ipynb`
- **모델**: U-Net (ResNet34, ImageNet pretrained)
- **태스크**: Binary segmentation (결절 vs 배경)
- **데이터**: TN3K — 3,493장 (Train 80% / Val 20%), 입력 256×256
- **불균형**: BG:Nodule = **~20:1** (중간 수준, 경계 불명확이 주요 난이도)
- **모달리티 특이점**: 초음파 특유의 speckle 노이즈, 불명확한 병변 경계
- **전처리**: BGR→RGB, 마스크 이진화(임계값 128), ImageNet 정규화
- **실험 Loss**: `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`, `cb_dice`
- **Optuna alpha 범위**: PLWCE 2.5~15.0, PWCE 0.2~2.5
- **평가 지표**: Dice, Sensitivity, Specificity, AUC
- **SoTA 참고** (2025년 1월 인터넷 조사 기준):
  | 방법 | Dice | IoU | Sensitivity | AUC | 출처 |
  |------|------|-----|-------------|-----|------|
  | MRDB (Mamba+ResNet, 2024) | **90.02%** | **81.85%** | 89.11% | — | PMC/Bioengineering |
  | ResUNet (2025) | 84.24% | 75.48% | 88.98% | — | AIMS Medical Science |
  | DPAM-UNet++ (2024) | 83.10% | 74.51% | 87.02% | 0.9213 | BMC Medical Imaging |
  | TRFE-Net / TRFE+ (2021/2022) | ~83% | ~71% | — | — | ISBI'21 / CBM'22 |
  | U-Net (ResNet34) baseline | ~78~82% | — | — | — | 복수 논문 |
- **데이터 로드**:
  - Google Drive: `MyDrive/imbalanced-data-LWCE/tn3k/` → `/tmp/tn3k_data/` 자동 복사
  - 로컬 fallback: `/root/imbalanced-data-LWCE/Thyroid Dataset/` (이미 존재 시 자동 사용)
  - 구조: `tg3k/thyroid-image/*.jpg` + `tn3k/tn3k-trainval-fold0.json`
- **결과 저장**: `results/tn3k_*.json/xlsx/png`

---

### 4-0-B. 뇌 백질 고강도 병변 MRI (WMH 2017)

- **파일**: `WMH_Brain_Lesion_MRI.ipynb`
- **모델**: U-Net (ResNet34, ImageNet pretrained)
- **태스크**: Binary segmentation (WMH 병변 vs 배경)
- **데이터**: WMH 2017 Challenge — 60개 케이스, 3개 병원(Utrecht/Singapore/Amsterdam)
- **불균형**: BG:WMH = **~100:1 ~ 600:1** (LiTS Tumor 356:1 초과, 극심한 불균형)
- **모달리티 특이점**: MRI 멀티채널 (FLAIR primary + T1 auxiliary)
- **입력 구성**: 3채널 `[FLAIR, T1, FLAIR]` → ImageNet 인코더 3ch 맞춤
- **전처리**: Percentile 정규화 (1~99th, foreground 기준), 볼륨 단위 Train/Val 분할
- **BG-only 슬라이스**: WMH 슬라이스 수의 30% 포함 (`BG_ONLY_RATIO=0.3`)
- **실험 Loss**: `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`, `cb_dice`
- **Optuna alpha 범위**: PLWCE **2.5~20.0** (극심한 불균형으로 범위 확장), PWCE 0.2~3.0
- **평가 지표**: Dice, Sensitivity, Specificity, AUC
- **SoTA 참고** (2026년 3월 인터넷 조사 기준):
  | 방법 | Dice | Sensitivity | Specificity | 출처 |
  |------|------|-------------|-------------|------|
  | Robust-WMH-UNet (2026) | **0.768** | — | — | arXiv |
  | Transformer-based (2025) | 0.720 | — | — | NeuroImage |
  | nnU-Net 3D (2021) | ~0.800* | — | — | Nature Methods (*Utrecht site) |
  | 2D U-Net | ~0.750~0.790 | — | — | WMH Challenge |
  > ⚠️ nnU-Net 0.800은 Utrecht 단일 사이트 기준; 3-site 평균은 낮을 수 있음
- **데이터 로드**:
  - **Kaggle (권장)**: `kagglehub.dataset_download("farahmo/wmh-dataset")` — 자동 다운로드
  - 공식 사이트: https://wmh.isi.uu.nl/ (현재 접속 불가 — kagglehub 사용 권장)
  - 구조: `{SiteName}/{SubjectID}/pre/FLAIR.nii.gz + T1.nii.gz` + `../wmh.nii.gz`
- **결과 저장**: `results/wmh_*.json/xlsx/png`
- **연구 의의**: 이 프로젝트에서 가장 극단적인 불균형 → LWCE 계열 효과 검증의 핵심 도메인

---

### 4-0-C. 조직병리 세포핵 분할 (MoNuSeg 2018)

- **파일**: `MoNuSeg_Nuclei_Pathology.ipynb`
- **모델**: U-Net (ResNet34, ImageNet pretrained)
- **태스크**: Binary segmentation (세포핵 vs 배경/세포질)
- **데이터**: MoNuSeg 2018 — 30 train + 14 test (TCGA 다기관 H&E 슬라이드)
- **불균형**: BG:Nucleus = **~3:1 ~ 8:1** (완만한 불균형, 경계 정밀도가 주요 난이도)
- **모달리티 특이점**: RGB H&E 병리 슬라이드 (CT/MRI/초음파와 완전히 다른 도메인)
- **전처리**: 패치 추출 (1000×1000 → 256×256 패치, stride=128, 50% overlap)
- **어노테이션**: XML polygon → 이진 마스크 자동 변환 (lxml 사용)
- **증강**: 좌우/상하 flip, 90° 회전, H&E 색상 증강 (밝기/대비 ±15%)
- **실험 Loss**: `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`, `cb_dice`
- **Optuna alpha 범위**: PLWCE 2.5~15.0, PWCE 0.2~2.5
- **평가 지표**: Dice, Sensitivity, Specificity, AUC
- **SoTA 참고** (2026년 3월 인터넷 조사 기준):
  | 방법 | Dice | AJI | 출처 |
  |------|------|-----|------|
  | FrGNet (2025) | — | **~0.640+** | arXiv (+2% vs HoverNet) |
  | FRE-Net | **0.856** | 0.628 | Medical Image Analysis |
  | HoverNet (2019) | 0.826 | 0.618 | Medical Image Analysis |
  | U-Net baseline | ~0.790~0.800 | ~0.580 | 복수 논문 |
- **데이터 로드**:
  - Google Drive: `MyDrive/imbalanced-data-LWCE/monuseg/` → `/tmp/monuseg_raw/` 자동 복사
  - 공식 사이트: https://monuseg.grand-challenge.org/
  - 구조: `MoNuSeg Training Data/Tissue Images/*.tif` + `Annotations/*.xml`
- **결과 저장**: `results/monuseg_*.json/xlsx/png`
- **연구 의의**: 완전히 새로운 병리 모달리티 추가, 완만한 불균형에서의 LWCE 효과 측정



### 4-0-D. SoTA 수치 작성 규칙

새로운 도메인을 추가하거나 SoTA 수치를 기재할 때 반드시 아래를 준수:

1. **인터넷 조사 필수**: SoTA 수치는 반드시 논문 검색 또는 Papers With Code, PubMed, arXiv 등을 통해 최신 수치 확인 후 기재
2. **조사 일자 명시**: `(YYYY년 MM월 인터넷 조사 기준)` 형식으로 조사 시점 표기
3. **출처 명시**: 방법명, 발표 연도, 출처(저널/학회/URL) 함께 기재
4. **표 형식 사용**: 단일 수치 대신 복수 방법 비교 표로 작성
5. **갱신 주기**: 실험 시작 전 6개월 이상 지난 SoTA는 재조사 권장

> ⚠️ 논문이나 GitHub README의 수치를 그대로 복사하지 말 것 — 데이터셋 split, 평가 방식이 다를 수 있음. 가능하면 공식 test set 기준 수치 사용.

---

### 4-1. 신규 도메인 선택 기준
새로운 의료 도메인을 추가할 때 반드시 아래 조건을 고려:
1. **불균형 비율**: 기존 도메인과 다른 불균형 패턴 (클래스 수, BG:FG 비율)
2. **이미지 모달리티**: 내시경(RGB), 안저(RGB), CT(Grayscale) 이후 다양화 권장
3. **공개 데이터셋**: 재현 가능한 벤치마크 데이터셋 사용
4. **기존 SoTA 모델 존재 여부**: 비교 기준점 확보

### 4-2. 추천 신규 도메인 후보

| 도메인 | 데이터셋 | 모달리티 | 클래스 수 | 특이점 |
|--------|----------|----------|-----------|--------|
| 피부 병변 | ISIC 2018 | Dermoscopy(RGB) | Binary | 형태 다양성 큼 |
| 폐 결절 | Luna16 / LIDC | CT 3D | Binary | 3D 볼륨 세그멘테이션 |
| 뇌 종양 | BraTS | MRI (4 modality) | 4-class | 다중 모달리티 |
| 유방암 | CBIS-DDSM | Mammography | Binary | 저해상도, 극심한 불균형 |
| 심장 구조 | ACDC | Cardiac MRI | 4-class | 시계열(동영상) |

---

## 5. 실험 설계 규칙 (반드시 준수)

### 5-1. 손실 함수 비교 규칙
- **기준선 필수 포함**: `ce_dice`는 항상 비교군에 포함
- **최소 비교 손실**: `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice` 4종 이상
- **alpha 탐색**: `plwce`, `pwce` 사용 시 반드시 Optuna로 최적 alpha 탐색 후 최종 실험

### 5-2. 데이터 분할 규칙
- **고정 시드**: `random_state=42` 고정 (재현성 보장)
- **분할 전략**: 공식 test split이 있으면 우선 사용, 없으면 **8:1:1 (Train/Val/Test)** 3분할
  - Val: 학습 중 Best Dice 체크포인트 기준 (early stopping)
  - Test: 학습 완료 후 최종 성능 보고 전용 (학습에 일절 관여 안 함)
- **단위**: 이미지/케이스/볼륨 단위로 분할 (슬라이스/패치 단위 금지 — data leakage 방지)

#### 도메인별 Test Set 현황

| 도메인 (노트북명) | Test Set 방식 | 비고 |
|--------|-------------|------|
| TN3K_Thyroid_Ultrasound | 공식 test split (`tn3k/test-image/`, `tn3k/test-mask/`) | fold JSON으로 train/val, 별도 test |
| Pancreas_MultiOrgan_CT | 공식 test volumes (`.h5`) | 이미 구현 완료 |
| MoNuSeg_Nuclei_Pathology | 공식 14개 test 케이스 (`MoNuSegTestData/`) | test 없으면 val fallback |
| Skin_Lesion_ISIC2018 | 랜덤 8:1:1 분할 | 공식 test GT 비공개 |
| LiTS_Liver_Tumor | 볼륨 단위 8:1:1 분할 | 공식 test GT 비공개 |
| WMH_Brain_Lesion_MRI | 케이스 단위 8:1:1 분할 | 공식 test GT 비공개 |
| Endoscopic_Polyp_Image | 랜덤 8:1:1 분할 | 공식 test split 없음 |
| Retinal_Image | Val set 사용 | 공식 test에 GT 없음 (불가피) |

### 5-3. 클래스 비율 계산 규칙
- 학습 데이터 전체에 대해 픽셀 단위로 계산
- `class_counts` 리스트로 저장 후 `get_loss_function()`에 전달
- 비율 출력 필수: `print(f"BG:FG = {ratio:.1f}:1")`

### 5-4. 모델 저장 규칙
- Best Val Dice 기준으로만 저장
- 학습 중 체크포인트(`.pth`) 및 슬라이스 캐시(`.npz`)는 `/tmp/`에 저장 (대용량 임시 파일)
- **최종 실험 결과는 `medical_data/results/{도메인}/`에 저장** (도메인별 하위 폴더)
  - `{domain}_optuna_results.json` — Optuna alpha 탐색 결과
  - `{domain}_optuna_search.png` — alpha 탐색 시각화
  - `{domain}_training_curves.png` — 학습 곡선
  - `{domain}_prediction_vis.png` — 예측 결과 시각화
  - `{domain}_final_results.json` — 최종 평가 지표 JSON
  - `{domain}_final_results.xlsx` — 최종 평가 지표 Excel (Summary + Training_History 시트)
  - `{domain}_final_metrics.png` — Loss별 최종 지표 바차트

#### 결과 폴더 구조

```
medical_data/results/
  TN3K_Thyroid_Ultrasound/    ← TN3K_Thyroid_Ultrasound.ipynb
  Skin_Lesion_ISIC2018/       ← Skin_Lesion_ISIC2018.ipynb
  LiTS_Liver_Tumor/           ← LiTS_Liver_Tumor.ipynb
  WMH_Brain_Lesion_MRI/       ← WMH_Brain_Lesion_MRI.ipynb
  MoNuSeg_Nuclei_Pathology/   ← MoNuSeg_Nuclei_Pathology.ipynb
  Pancreas_MultiOrgan_CT/     ← Pancreas_MultiOrgan_CT.ipynb
  Retinal_Image/              ← Retinal_Image.ipynb
  Endoscopic_Polyp_Image/     ← Endoscopic_Polyp_Image.ipynb
```

- 각 노트북 Cell 0 또는 Cell 1에 아래 형식으로 설정:
  ```python
  RESULTS_DIR = '/root/imbalanced-data-LWCE/medical_data/results/{도메인}'
  os.makedirs(RESULTS_DIR, exist_ok=True)
  ```

### 5-5. 평가 지표 규칙

| 태스크 | 필수 지표 | 선택 지표 |
|--------|-----------|-----------|
| Binary seg | Dice, Sensitivity, Specificity | AUC, IoU |
| Multi-class seg | 클래스별 Dice, mDice (BG 제외) | HD95 |
| 3D 볼륨 | 슬라이스 단위 예측 → 볼륨 단위 집계 | NSD |

---

## 6. 코드 작성 규칙

### 6-1. import 및 경로

```python
# custom_losses 경로 (올바른 경로 사용 — 'imbalanced' 주의)
sys.path.insert(0, '/root/imbalanced-data-LWCE/medical_data')
from custom_losses import get_loss_function, calculate_weights
```

> **주의**: 기존 노트북에 `/root/inbalanced-data-LWCE/` (오타) 존재 → 신규 코드에서는 반드시 `imbalanced`로 수정

### 6-2. Binary 세그멘테이션 1채널 출력 처리

```python
# sigmoid 1채널 출력 → 2채널 logit 변환 (핵심 버그 수정)
def to_2ch_logits(p):
    return torch.cat([-p, p], dim=1)
```

- `[zeros, p]` 방식 금지 → 배경 logit이 0으로 고정되어 학습 불안정
- `[-p, p]` 방식 사용 → sigmoid(p)와 수학적으로 동일

### 6-3. 가중치 device 동기화

```python
# forward 내에서 device 불일치 방지
if self.weights is not None and self.weights.device != logits.device:
    self.weights = self.weights.to(logits.device)
```

### 6-4. Optuna 설정 표준

```python
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
)
# proxy 설정: subset_ratio=0.15, epochs=5
# N_TRIALS     = 20~30   (alpha 단일 파라미터 탐색)
# N_TRIALS_PF  = N_TRIALS * 2  (PLWCE+Focal: alpha+gamma 2개 파라미터 → 탐색 공간 2배)
```

#### 도메인별 Optuna 탐색 범위

| 도메인 | 불균형 | PLWCE α | PWCE α | PLWCE+Focal α | PLWCE+Focal γ | N_TRIALS | N_TRIALS_PF |
|--------|--------|---------|--------|---------------|---------------|----------|-------------|
| Skin_Lesion_ISIC2018 | 중간 | 2.5~15.0 | 0.2~2.5 | 2.5~15.0 | 0.5~5.0 | 20 | 40 |
| TN3K_Thyroid_Ultrasound | ~20:1 | 2.5~15.0 | 0.2~2.5 | 2.5~15.0 | 0.5~5.0 | 20 | 40 |
| MoNuSeg_Nuclei_Pathology | ~3~8:1 | 2.5~15.0 | 0.2~2.5 | 2.5~15.0 | 0.5~5.0 | 20 | 40 |
| WMH_Brain_Lesion_MRI | ~100~600:1 | **2.5~20.0** | 0.2~3.0 | **2.5~20.0** | 0.5~5.0 | 30 | 60 |
| Pancreas_MultiOrgan_CT | 극심 (췌장) | 2.5~15.0 | 0.2~2.5 | 2.5~15.0 | 0.5~5.0 | 30 | 60 |

> WMH는 극심한 불균형으로 PLWCE α 상한을 15.0 → **20.0** 으로 확장.
> Pancreas는 다중 클래스라 trial 수를 30/60으로 늘림 (수렴 느림).

### 6-5. 시각화 규칙
- `matplotlib.use('Agg')` 서버 환경에서 필수
- 학습 곡선 (Loss + Val Metric) 반드시 저장
- 예측 시각화: Input / GT / Prob Map / Pred 4열 구성

### 6-6. 실험 환경 구조 (VS Code + Colab + Google Drive)

#### 환경 구조 이해

```
Google Drive  ←  데이터 영구 보관
      ↑
  Colab mount (브라우저에서 인증)
      ↑
VS Code Remote Tunnel  ←  코드 편집 클라이언트
```

- **VS Code**는 단순 편집 클라이언트. 실제 실행은 Colab 서버에서 이루어짐
- **Drive 마운트는 반드시 Colab에서** 수행 (VS Code에서 직접 마운트 불가)
- 마운트 후 VS Code로 접속하면 `/content/drive/MyDrive` 경로가 그대로 보임

#### 올바른 작업 순서

```
1. 브라우저에서 Colab 노트북 열기
2. 첫 셀 실행 → drive.mount('/content/drive') → 구글 계정 인증
3. VS Code Remote Tunnel로 연결
4. 이후 모든 작업은 VS Code에서 진행
```

#### Google Drive 데이터 로딩 패턴 (표준)

직접 다운로드 불가 데이터셋은 아래 표준 패턴 사용.

**Cell 0 — Drive 마운트 + 경로 변수**

```python
GDRIVE_DATA_PATH = '/content/drive/MyDrive/imbalanced-data-LWCE/{domain}'

IS_COLAB = False
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IS_COLAB = True
    print('Google Drive 마운트 완료')
except Exception:
    print('Colab 환경 아님 — 로컬/수동 경로 사용')
```

**Cell 1 — Drive → /tmp 복사 (빠른 I/O)**

Drive는 네트워크 파일시스템(FUSE)이라 파일 I/O마다 네트워크 왕복 발생.
→ 학습 시작 전 `/tmp`(로컬 SSD)로 한 번만 복사해두면 이후 I/O는 로컬 속도로 동작.

```python
DATA_DIR = '/tmp/{domain}_raw'
if IS_COLAB and os.path.exists(GDRIVE_DATA_PATH):
    import shutil
    if not os.path.exists(os.path.join(DATA_DIR, '{check_subdir}')):
        print('Google Drive에서 데이터 복사 중...')
        shutil.copytree(GDRIVE_DATA_PATH, DATA_DIR, dirs_exist_ok=True)
else:
    if not os.path.exists(DATA_DIR):
        print('[데이터 없음] Drive 경로 확인 또는 /tmp에 직접 배치')
```

#### 파일 저장 전략

```
Google Drive (영구 보관)          /tmp (학습 중 임시)
  └── results/*.json/xlsx/png      └── {domain}_raw/     ← 원본 데이터 복사본
                                   └── {domain}_slices/  ← 전처리 캐시
                                   └── best_*.pth        ← 체크포인트
```

- 학습 결과(`results/`)는 `RESULTS_DIR = '/root/imbalanced-data-LWCE/medical_data/results'`에 저장
- Colab 런타임 종료 시 `/tmp` 내용은 모두 삭제됨 → 결과 파일은 반드시 `results/`에 저장

#### Google Drive 업로드 폴더 구조 (전 도메인 통일)

```
MyDrive/imbalanced-data-LWCE/
  synapse/     train_npz/                      ← Pancreas CT (Synapse)
               test_vol_h5/
  wmh/         {SiteName}/{SubjectID}/pre/      ← WMH 2017 MRI
               FLAIR.nii.gz, T1.nii.gz
               ../wmh.nii.gz
  monuseg/     MoNuSeg Training Data/           ← MoNuSeg 2018 Pathology
               Tissue Images/*.tif
               Annotations/*.xml
  tn3k/        tg3k/thyroid-image/*.jpg         ← TN3K Thyroid Ultrasound
               tg3k/thyroid-mask/*.jpg
               tn3k/test-image/*.jpg
               tn3k/tn3k-trainval-fold0.json
```

#### 적용 노트북 현황

| 노트북 | Drive 경로 키 | /tmp 복사 대상 | 상태 |
|--------|--------------|---------------|------|
| `Pancreas_MultiOrgan_CT.ipynb` | `synapse` | `/tmp/synapse_data` | ✅ 완료 |
| `WMH_Brain_Lesion_MRI.ipynb` | `wmh` | `/tmp/wmh_raw` | ✅ 완료 |
| `MoNuSeg_Nuclei_Pathology.ipynb` | `monuseg` | `/tmp/monuseg_raw` | ✅ 완료 |
| `TN3K_Thyroid_Ultrasound.ipynb` | `tn3k` | `/tmp/tn3k_data` | ✅ 완료 (로컬 fallback 포함) |

> **TN3K 특이점**: 로컬 환경에서는 `/root/imbalanced-data-LWCE/Thyroid Dataset/`이 이미 존재하므로 Drive 마운트 없이 자동으로 로컬 경로 사용.

---

## 7. 알려진 이슈 및 주의사항

| 이슈 | 원인 | 해결책 |
|------|------|--------|
| `ModuleNotFoundError: optuna` | 환경에 optuna 미설치 | `!pip install optuna` |
| 경로 오타 `inbalanced` | 초기 노트북 오타 | 신규 파일은 `imbalanced` 사용 |
| DRIVE test에 GT 없음 | 데이터셋 특성 | val set으로 평가 후 명시 |
| TransUNet ViT 가중치 다운로드 필요 | 외부 URL | 사전 다운로드 또는 캐시 확인 |
| PraNet 상대 import 오류 | 외부 레포 클론 후 패치 필요 | `from .Res2Net_v1b` → `from Res2Net_v1b` |

---

## 8. 결과 기록 표준 (논문/보고서용)

```
실험 결과 표 형식:
| Loss Function    | alpha | gamma | Dice | Sens | Spec | AUC |
|------------------|-------|-------|------|------|------|-----|
| CE+Dice          | -     | -     |      |      |      |     |
| WCE+Dice         | -     | -     |      |      |      |     |
| LWCE+Dice        | -     | -     |      |      |      |     |
| PLWCE+Dice       | best  | -     |      |      |      |     |
| CB+Dice          | -     | -     |      |      |      |     |
| PLWCE+Focal+Dice | best  | best  |      |      |      |     |
```

- alpha, gamma는 Optuna 최적값 기재 (단일 파라미터 모드는 해당 없는 항목 `-` 표시)
- 최고 성능 수치 **볼드** 처리
- 불균형 비율(BG:FG)도 함께 기재

### 8-1. 최종 논문 결과 기록 방식 — 반복 실험 (구현 보류 중)

> ⚠️ **구현 시점**: 모든 실험 도메인이 확정된 이후 일괄 적용. 현재는 설계만 기록.

손실 함수 간 성능 차이가 작을 수 있으므로, 단일 실행 결과가 아닌 **반복 실험의 평균 및 표준편차**를 최종 논문 수치로 사용한다.

#### 실험 프로토콜
- **반복 횟수**: 5회 (seed=0, 1, 2, 3, 4)
- **각 반복**: 동일한 최적 alpha/gamma로 전체 학습 실행 (Optuna 탐색은 1회만 수행, 이후 고정)
- **기록 지표**: 각 Loss × 5회의 Dice (또는 mDice), Sensitivity, Specificity, AUC
- **최종 수치**: `mean ± std` 형식

#### 최종 결과 표 형식 (논문용)

```
| Loss Function    | alpha | gamma | Dice (mean±std)      | Sens (mean±std) | Spec (mean±std) |
|------------------|-------|-------|----------------------|-----------------|-----------------|
| CE+Dice          | -     | -     | 0.XXX ± 0.XXX        |                 |                 |
| WCE+Dice         | -     | -     | 0.XXX ± 0.XXX        |                 |                 |
| LWCE+Dice        | -     | -     | 0.XXX ± 0.XXX        |                 |                 |
| PLWCE+Dice       | best  | -     | **0.XXX ± 0.XXX**    |                 |                 |
| CB+Dice          | -     | -     | 0.XXX ± 0.XXX        |                 |                 |
| PLWCE+Focal+Dice | best  | best  | 0.XXX ± 0.XXX        |                 |                 |
```

#### 평균 랭킹 분석
- 모든 도메인에 대해 각 Loss의 Dice 순위를 매기고 Borda count로 종합 순위 산출
- 표 형식: 행=Loss Function, 열=도메인, 셀=해당 도메인에서의 랭킹
- 타뷸라 데이터 실험(KIIS 발표)과 동일한 히트맵 시각화 방식 적용

#### 구현 위치
- 각 노트북의 Cell 6(평가/저장) 뒤에 **Cell 7: 반복 실험** 셀을 추가
- 공통 유틸리티 함수는 `medical_data/repeat_experiment.py`로 분리 예정

---

## 9. 신규 도메인 노트북 체크리스트

새로운 도메인 실험 노트북 작성 시 반드시 아래 순서로 구성:

- [ ] Cell 0: 라이브러리 import + 경로 설정 + device 설정
- [ ] Cell 1: 데이터셋 로드 + Dataset 클래스 + DataLoader
- [ ] Cell 2: **클래스 비율 계산** (픽셀 단위) + `class_counts` 출력
- [ ] Cell 3: 모델 정의 + 유틸리티 함수 (val metric 계산)
- [ ] Cell 4: 학습 함수 (`train_{model}`) 정의
- [ ] Cell 5: **(옵션)** Optuna alpha 탐색 (`plwce`, `pwce` 사용 시 필수)
- [ ] Cell 6: 전체 Loss 비교 학습 실행
- [ ] Cell 7: 시각화 (학습 곡선 + 예측 결과)
- [ ] Cell 8: 최종 평가 지표 출력 + JSON 저장

---

## 10. 노트북 통일 구조 표준

모든 노트북은 아래 구조를 완전히 동일하게 유지한다. 기존 노트북도 이 표준으로 통일할 것.

### 10-1. 셀 구성 순서 (표준)

```
[Markdown] 노트북 헤더 (제목, 도메인, 모달리티, 모델, 데이터셋 설명)
[Code]     Cell 0 — 환경 설정 (pip install, import, 경로, device)
[Code]     Cell 1 — 데이터 로드 (Drive/kagglehub 마운트, /tmp 복사, Dataset, DataLoader, 클래스비율)
[Code]     Cell 2 — 모델 정의 (build_model, to_2ch_logits 등 유틸리티)
[Code]     Cell 3 — 학습 함수 (train_{model} 정의 — 실행 안 함)
[Code]     Cell 4 — Optuna alpha/gamma 탐색 (PLWCE, PWCE, PLWCE+Focal)
[Code]     Cell 5 — 전체 Loss 비교 학습 실행
[Code]     Cell 6 — 평가 및 결과 저장 (시각화, JSON, Excel, PNG)
```

- **Markdown 헤더 셀**은 Cell 0 앞에 반드시 위치 (모든 노트북 통일)
- Cell 번호는 0-based (Cell 0, Cell 1, …)
- **각 셀 첫 줄에 `# === Cell N: 제목 ===` 형식의 주석 추가** (Python 코드 셀 기준)

### 10-2. Markdown 헤더 셀 형식 (표준)

노트북을 처음 여는 사람(협업자, 리뷰어)이 코드를 한 줄도 읽지 않고도 **무엇을, 왜, 어떻게** 하는지 파악할 수 있어야 한다.
아래 7개 섹션을 모두 포함할 것.

```markdown
# {도메인 이름} 불균형 세그멘테이션 실험

---

## 1. 태스크 및 도메인
- **도메인**: {예: 갑상선 결절 초음파 세그멘테이션}
- **모달리티**: {CT / MRI (FLAIR+T1) / Ultrasound / Dermoscopy / H&E Pathology / Fundus}
- **태스크**: {Binary / 3-class / 9-class} segmentation
  - 클래스: {예: 배경(0) / 결절(1)} 또는 {배경 + 8개 장기}
- **핵심 도전**: {예: 경계 불명확, speckle 노이즈, 극심한 불균형, 소형 병변 등}

## 2. 모델
- **아키텍처**: {예: U-Net (ResNet34 백본) / TransUNet (R50+ViT-B/16) / PraNet (Res2Net50)}
- **사전학습**: {예: ImageNet pretrained / ViT-B/16 pretrained}
- **선택 이유**: {예: Binary segmentation 표준 베이스라인 / 멀티스케일 특성 필요 / Transformer 장거리 의존성}
- **출력**: {예: 1채널 sigmoid → to_2ch_logits() 변환 / 9채널 softmax}

## 3. 데이터셋
- **이름**: {예: TN3K (Thyroid Nodule 3K)}
- **규모**: {예: 3,493장 — Train 80% / Val 20% / Test: 공식 별도 제공}
- **입력 해상도**: {예: 256×256 (리사이즈)}
- **클래스 불균형**: BG:FG = **{비율}** ({불균형 수준 — 완만/중간/심각/극심})
- **공식 분할**: {예: fold JSON 제공 / 볼륨 단위 공식 train-test 분리 / 없음 → 8:1:1 랜덤}

## 4. 데이터 준비 (협업자용)
> 아래 방법으로 데이터를 준비한 후 노트북을 실행할 것.

**취득 방법**:
- {예1: `kagglehub.dataset_download("farahmo/wmh-dataset")` — Cell 0 실행 시 자동}
- {예2: https://monuseg.grand-challenge.org/ 계정 등록 후 수동 다운로드}

**Colab Drive 업로드 경로** (수동 다운로드 시):
```
MyDrive/imbalanced-data-LWCE/{domain}/
  {폴더 구조 기재}
```

**로컬 경로** (로컬 실행 시):
- {예: `/root/imbalanced-data-LWCE/Thyroid Dataset/` 에 위치 시 자동 감지}

## 5. 전처리 및 도메인 특이점
- {예: CLAHE 적용 (Green 채널, clipLimit=2.0) — 혈관 대비 강조}
- {예: 입력 3채널 구성 [FLAIR, T1, FLAIR] — ImageNet 인코더 3ch 맞춤}
- {예: BG-only 슬라이스 30% 포함 (BG_ONLY_RATIO=0.3)}
- {예: 패치 추출 1000×1000 → 256×256, stride=128 (50% overlap)}
- {예: BGR→RGB 변환, 마스크 이진화 임계값 128}

## 6. 실험 손실 함수 및 Optuna 탐색 범위
| 손실 함수 | 탐색 파라미터 | 탐색 범위 | Trials |
|-----------|-------------|----------|--------|
| `ce_dice` | — | — | — |
| `wce_dice` | — | — | — |
| `lwce_dice` | — | — | — |
| `plwce_dice` | alpha | {예: 2.5 ~ 15.0} | {N_TRIALS} |
| `pwce_dice` | alpha | {예: 0.2 ~ 2.5} | {N_TRIALS} |
| `cb_dice` | — | — | — |
| `plwce_focal_dice` | alpha + gamma | alpha {범위}, gamma {0.5 ~ 5.0} | {N_TRIALS_PF} |

## 7. SoTA 참고 ({YYYY년 MM월} 기준)
| 방법 | {주요 지표} | {보조 지표} | 출처 |
|------|------------|------------|------|
| {최신 SOTA 방법} | **X.XXX** | X.XXX | {저널/학회} |
| {비교 방법 2} | X.XXX | X.XXX | {출처} |
| U-Net baseline | X.XXX | — | 복수 논문 |

> 본 연구 목표: U-Net baseline 대비 LWCE/PLWCE 계열 손실 함수의 개선 효과 검증.
> 평가 지표: {예: Dice, Sensitivity, Specificity, AUC}
> 결과 저장: `medical_data/results/{도메인폴더}/`
```

#### 작성 원칙
- **섹션 4 (데이터 준비)**는 반드시 실제 동작하는 경로/명령어로 작성 — 추상적 설명 금지
- **섹션 5 (전처리)**는 코드를 보지 않아도 이해할 수 있게 — "왜" 이 전처리를 하는지 포함
- **섹션 7 (SoTA)**는 rules.md §3, §4의 표를 그대로 복사해서 넣을 것 (별도 조사 불필요)
- Markdown 헤더는 **읽는 문서**이지 코드 요약본이 아님 — 코드에 이미 있는 내용 단순 반복 금지

### 10-3. 도메인별 데이터 준비 방법 (협업자용)

GitHub에서 클론 후 각 도메인 데이터를 아래 방법으로 준비한다.

| 도메인 | 파일 | 데이터 취득 방법 | Colab 업로드 경로 |
|--------|------|----------------|-----------------|
| WMH 2017 MRI | `WMH_Brain_Lesion_MRI.ipynb` | **자동**: `kagglehub.dataset_download("farahmo/wmh-dataset")` — Cell 0 실행 시 자동 다운로드 | 불필요 |
| TN3K Thyroid | `TN3K_Thyroid_Ultrasound.ipynb` | [TN3K GitHub](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) 또는 직접 다운로드 | `MyDrive/imbalanced-data-LWCE/tn3k/` |
| MoNuSeg 2018 | `MoNuSeg_Nuclei_Pathology.ipynb` | [공식 챌린지](https://monuseg.grand-challenge.org/) 계정 등록 후 다운로드 | `MyDrive/imbalanced-data-LWCE/monuseg/` |
| Skin ISIC 2018 | `Skin_Lesion_ISIC2018.ipynb` | `kagglehub.dataset_download(...)` — Cell 0 실행 시 자동 (또는 [ISIC Archive](https://challenge.isic-archive.com/)) | 자동 |
| Pancreas Synapse | `Pancreas_MultiOrgan_CT.ipynb` | [Synapse Platform](https://www.synapse.org/#!Synapse:syn3193805/wiki/) 계정 등록 → zip 압축 후 업로드 | `MyDrive/imbalanced-data-LWCE/synapse/synapse.zip` |

#### 상세 폴더 구조 (Drive 업로드 기준)

```
MyDrive/imbalanced-data-LWCE/
  synapse/
    synapse.zip              ← train_npz/ + test_vol_h5/ 압축 (전체 업로드)
  tn3k/
    tg3k/
      thyroid-image/*.jpg    ← TG3K 갑상선 이미지
      thyroid-mask/*.jpg     ← TG3K 갑상선 마스크
    tn3k/
      trainval-image/*.jpg   ← TN3K 훈련/검증 이미지
      trainval-mask/*.jpg    ← TN3K 훈련/검증 마스크
      test-image/*.jpg       ← TN3K 테스트 이미지
      test-mask/*.jpg        ← TN3K 테스트 마스크
      tn3k-trainval-fold0.json
  monuseg/
    MoNuSeg Training Data/
      Tissue Images/*.tif
      Annotations/*.xml
    MoNuSegTestData/
      Tissue Images/*.tif
      Annotations/*.xml
  wmh/                       ← WMH는 kagglehub 자동 다운로드, Drive 불필요
```

### 10-4. 셀 내부 주석 스타일 (표준)

```python
# === Cell 0: 환경 설정 ===

# --- 라이브러리 설치 ---
# !pip install segmentation-models-pytorch albumentations optuna kagglehub

# --- 라이브러리 import ---
import os, sys, json
...

# --- 경로 및 하이퍼파라미터 설정 ---
DOMAIN = 'skin_lesion'      # 결과 파일명 prefix
IMG_SIZE = 256
...
```

- 섹션 구분: `# --- 소제목 ---`
- 셀 헤더: `# === Cell N: 셀 이름 ===`
- 인라인 설명: 코드 오른쪽에 짧게 (`# 배경 포함 전체 클래스 수`)
- 경고/주의: `# ⚠️ 주의: ...` 형식

### 10-5. 현재 노트북 구조 적합성 현황

| 노트북 | Markdown 헤더 | Cell 번호 표준 | PLWCE+Focal | Optuna TQDM_DISABLE | study_pf 순서 |
|--------|:------------:|:------------:|:-----------:|:------------------:|:-----------:|
| `Skin_Lesion_ISIC2018.ipynb` | ✅ | 🔲 미확인 | 🔲 미추가 | 🔲 미확인 | ✅ |
| `TN3K_Thyroid_Ultrasound.ipynb` | ✅ | 🔲 미확인 | 🔲 미추가 | ✅ | ✅ |
| `WMH_Brain_Lesion_MRI.ipynb` | 🔲 없음 | 🔲 미확인 | 🔲 미추가 | 🔲 미확인 | ✅ |
| `MoNuSeg_Nuclei_Pathology.ipynb` | 🔲 없음 | 🔲 미확인 | 🔲 미추가 | ✅ | ✅ |
| `Pancreas_MultiOrgan_CT.ipynb` | 🔲 없음 | 🔲 미확인 | 🔲 미추가 | ✅ | ✅ |

> **다음 작업**: rules.md 기준으로 위 체크리스트 완성. Markdown 헤더 없는 노트북 3개 추가, `plwce_focal_dice` 전 노트북 적용.

---

## 11. CLAUDE.md vs rules.md 차이

| 항목 | `CLAUDE.md` | `rules.md` |
|------|-------------|------------|
| 로드 방식 | Claude Code 세션 시작 시 **자동 로드** | `@medical_data/rules.md` 명시 참조 시에만 로드 |
| 용도 | Claude가 따라야 할 행동 지침, 코딩 스타일, 금지 사항 | 연구 규칙, 실험 설계, 도메인 정보 문서 |
| 대상 독자 | Claude Code AI 어시스턴트 | 연구팀 협업자 + Claude |
| 버전 관리 | 리포지토리 루트에 두면 git으로 관리됨 | git으로 관리됨 |
| 권장 위치 | `/root/imbalanced-data-LWCE/CLAUDE.md` (루트) | `medical_data/rules.md` (현재 위치) |

**권장 방향**: 현재 `rules.md`를 그대로 유지하면서, 루트에 `CLAUDE.md`를 별도로 만들어 두면 Claude Code가 자동으로 연구 컨텍스트를 인식함.

`CLAUDE.md`에 넣을 내용 (예시):
```markdown
# 프로젝트 컨텍스트
의료 이미지 세그멘테이션 불균형 연구 프로젝트.
자세한 규칙과 실험 설계는 @medical_data/rules.md 참조.

# 핵심 규칙 요약
- 손실 함수는 항상 get_loss_function()을 통해 생성
- to_2ch_logits(p) 패턴 사용 ([-p, p] 방식)
- Optuna로 alpha/gamma 탐색 후 최종 실험
- 결과는 medical_data/results/{도메인}/ 에 저장
```
