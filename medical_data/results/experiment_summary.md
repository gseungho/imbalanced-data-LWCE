# LWCE 실험 결과 요약

> **마지막 업데이트**: 2026-03-22
> 결과 기준: Test Set (val fallback은 별도 표기)
> 수치 기준: **Dice** (Binary: FG Dice / Multi-class: mDice)
> 표 범례: **굵은 수치** = 해당 열 최고값 | 🥇🥈🥉 = 본 연구 내 랭킹 | ⭐ SoTA 참고행

---

## 완료된 실험 현황

| 도메인 | 노트북 | 불균형 비율 | 태스크 | 모델 | 상태 |
|--------|--------|-----------|--------|------|------|
| Kvasir-SEG (Endoscopic Polyp) | `Endoscopic_Polyp_Image.ipynb` | ~5~10:1 (추정) | Binary | U-Net (ResNet34) | ✅ 완료 |
| DRIVE (Retinal Vessel) | `Retinal_Image.ipynb` | ~9:1 (추정) | Binary | IterNet (ResNet34) | ✅ 완료 |
| ISIC 2018 (Skin Lesion) | `Skin_Lesion_ISIC2018.ipynb` | **3.6:1** | Binary | U-Net (ResNet34) | ✅ 완료 |
| MoNuSeg 2018 (Nuclei) | `MoNuSeg_Nuclei_Pathology.ipynb` | **2.6:1** | Binary | U-Net (ResNet34) | ✅ 완료 |
| TN3K (Thyroid Nodule) | `TN3K_Thyroid_Ultrasound.ipynb` | **6.0:1** | Binary | U-Net (ResNet34) | ✅ 완료 |
| LiTS 2017 (Liver Tumor) | `LiTS_Liver_Tumor.ipynb` | BG:Liver **15:1**, BG:Tumor **356:1** | 3-class | U-Net++ (ResNet50) | ✅ 완료 |
| Pancreas/Synapse (Multi-organ) | `Pancreas_MultiOrgan_CT.ipynb` | 기관별 상이 | 8-class | TransUNet | ✅ 완료 |
| WMH 2017 (Brain Lesion) | `WMH_Brain_Lesion_MRI.ipynb` | **250:1** | Binary | U-Net (ResNet34) | ✅ 완료 |
| BUSI (Breast Ultrasound) | `BUSI_Breast_Ultrasound.ipynb` | — | Binary | U-Net (ResNet34) | 🔄 진행 중 |
| ACDC (Cardiac MRI) | `ACDC_Cardiac_MRI.ipynb` | BG:구조 ~9:1 (추정) | 4-class | U-Net (ResNet34) | 🔄 진행 중 |
| REFUGE 2018 (Optic Disc/Cup) | `REFUGE_Optic_Disc_Cup.ipynb` | Disc:Cup ~4:1 | 3-class | U-Net (ResNet34) | 🔄 진행 중 |

---

## 1. 도메인별 결과 (Dice 기준 랭킹)

---

### 1-1. Kvasir-SEG — Endoscopic Polyp Segmentation

| 모달리티 | 태스크 | 불균형 | 평가셋 | Train/Val 분할 |
|---------|--------|--------|--------|---------------|
| 내시경 RGB | Binary (BG/Polyp) | ~5~10:1 (추정) | Kvasir test | 80/20 random |

| 순위 | 방법 | Dice ↓ | AUC | 비고 |
|-----|------|--------|-----|------|
| ⭐ | MNet-SAt (2024) | **0.9661** | — | IEEE Access |
| ⭐ | ARCUNet (2025) | 0.9534 | — | arXiv |
| ⭐ | PraNet (2021) | 0.8980 | — | MICCAI'21 |
| ⭐ | U-Net baseline | ~0.79~0.82 | — | 복수 논문 |
| — | — | — | — | — |
| 🥇 1 | `lwce_dice` | **0.8993** | 0.9889 | |
| 🥈 2 | `ce_dice` | 0.8940 | **0.9908** | 기준선 |
| 🥉 3 | `cb_dice` | 0.8920 | 0.9905 | |
| 4 | `plwce_dice` | 0.8920 | 0.9902 | |
| 5 | `plwce_focal_dice` | 0.8893 | 0.9893 | |
| 6 | `wce_dice` | 0.8875 | 0.9904 | |

> **비고**: lwce_dice가 CE 대비 +0.005. PraNet(0.898)과 거의 동등. 전반적으로 손실함수 간 차이 미미.
> `pwce_dice` 미실험.

---

### 1-2. DRIVE — Retinal Vessel Segmentation

| 모달리티 | 태스크 | 불균형 | 평가셋 | 모델 |
|---------|--------|--------|--------|------|
| 안저 RGB | Binary (BG/Vessel) | ~9:1 (추정) | val 4장 (GT 없는 test 제외) | IterNet (ResNet34) |

| 순위 | 방법 | Dice ↓ | Sensitivity | Specificity | AUC | 비고 |
|-----|------|--------|------------|------------|-----|------|
| ⭐ | GAN+UNet (2024) | **0.8215** | 0.8301 | 0.9781 | 0.9772 | IEEE JBHI |
| ⭐ | LU-RA Transformer (2025) | 0.7871 | — | — | — | Applied Intelligence |
| ⭐ | IterNet 원본 (2020) | ~0.779 | — | — | ~0.979 | AAAI'20 |
| ⭐ | U-Net baseline | ~0.730 | — | — | ~0.974 | 복수 논문 |
| — | — | — | — | — | — | — |
| 🥇 1 | `plwce_focal_dice` | **0.6957** | 0.7502 | 0.9598 | 0.9510 | |
| 🥈 2 | `lwce_dice` | 0.6955 | 0.7315 | **0.9630** | 0.9487 | |
| 🥉 3 | `plwce_dice` | 0.6933 | 0.7476 | 0.9595 | 0.9496 | |
| 4 | `ce_dice` | 0.6911 | 0.7453 | 0.9591 | 0.9340 | 기준선 |
| 5 | `wce_dice` | 0.6548 | **0.7965** | 0.9368 | **0.9538** | |

> **비고**: 본 연구 결과(0.695~0.696)가 SoTA(0.82) 및 IterNet 원본(0.779) 대비 낮음.
> val 4장(GT 제공)으로만 평가 가능한 구조적 한계. `cb_dice`, `pwce_dice` 미실험.

---

### 1-3. ISIC 2018 — Skin Lesion Segmentation

| 모달리티 | 태스크 | 불균형 | 평가셋 | Train/Val/Test |
|---------|--------|--------|--------|---------------|
| 피부경 RGB | Binary (BG/Lesion) | **3.6:1** | 260장 test | 2075/259/260 |

| 순위 | 방법 | Dice ↓ | Sensitivity | Specificity | AUC | 비고 |
|-----|------|--------|------------|------------|-----|------|
| ⭐ | ARCUNet (2025) | **0.9534** | — | — | — | arXiv |
| ⭐ | Meta-UNet (2025) | 0.9314 | — | — | — | Applied Sciences |
| ⭐ | UNet++ (2019) | ~0.890 | — | — | — | MICCAI'18 |
| ⭐ | U-Net baseline | ~0.85~0.88 | — | — | — | 복수 논문 |
| — | — | — | — | — | — | — |
| 🥇 1 | `lwce_dice` | **0.9015** | 0.9137 | 0.9676 | 0.9937 | |
| 🥈 2 | `plwce_focal_dice` (α=2.09, γ=1.39) | 0.8984 | 0.9048 | **0.9725** | 0.9940 | |
| 🥉 3 | `cb_dice` | 0.8956 | 0.9158 | 0.9666 | 0.9938 | |
| 4 | `ce_dice` | 0.8950 | 0.9148 | 0.9663 | 0.9939 | 기준선 |
| 5 | `wce_dice` | 0.8941 | **0.9350** | 0.9545 | 0.9937 | |
| 6 | `plwce_dice` (α=12.72) | 0.8934 | 0.9163 | 0.9680 | **0.9945** | |

> **비고**: 최우수(lwce_dice 0.9015)가 UNet++ SoTA(~0.890) 상회. ARCUNet(0.953)과는 격차 존재.
> 낮은 불균형(3.6:1)에서도 lwce_dice 소폭 우세.

---

### 1-4. MoNuSeg 2018 — Nuclei Pathology Segmentation

| 모달리티 | 태스크 | 불균형 | 평가셋 | 패치 추출 |
|---------|--------|--------|--------|---------|
| H&E 병리 RGB | Binary (BG/Nucleus) | **2.6:1** | val (공식 test GT 미포함) | 256×256, stride 128 |

| 순위 | 방법 | Dice ↓ | Sensitivity | Specificity | AUC | 비고 |
|-----|------|--------|------------|------------|-----|------|
| ⭐ | FRE-Net | **0.856** | — | — | — | Medical Image Analysis |
| ⭐ | HoverNet (2019) | 0.826 | — | — | — | Medical Image Analysis |
| ⭐ | U-Net baseline | ~0.790~0.800 | — | — | — | 복수 논문 |
| — | — | — | — | — | — | — |
| 🥇 1 | `ce_dice` | **0.8168** | 0.8130 | **0.9497** | **0.9675** | 기준선 |
| 🥈 2 | `plwce_focal_dice` (α=2.68, γ=1.90) | 0.8161 | 0.8199 | 0.9463 | 0.9663 | |
| 🥉 3 | `wce_dice` | 0.8156 | **0.8580** | 0.9304 | 0.9669 | |
| 4 | `plwce_dice` (α=2.85) | 0.8126 | 0.8030 | 0.9509 | 0.9643 | |
| 5 | `lwce_dice` | 0.8081 | 0.7892 | 0.9536 | 0.9648 | |
| 6 | `cb_dice` | 0.7994 | 0.8193 | 0.9347 | 0.9613 | |

> **비고**: 가장 낮은 불균형(2.6:1) — 손실함수 간 차이 최대 0.017로 미미. CE 기준선이 최우수.
> U-Net baseline SoTA(~0.79~0.80) 상회. HoverNet(0.826) 대비 소폭 낮음.
> `pwce_dice` 미실험. val 기반 평가로 공식 test와 직접 비교 주의.

---

### 1-5. TN3K — Thyroid Nodule Ultrasound Segmentation

| 모달리티 | 태스크 | 불균형 | 평가셋 | Train/Val/Test |
|---------|--------|--------|--------|---------------|
| 초음파 그레이스케일 | Binary (BG/Nodule) | **6.0:1** | 614장 공식 test | 2303/576/614 |

| 순위 | 방법 | Dice ↓ | Sensitivity | Specificity | AUC | 비고 |
|-----|------|--------|------------|------------|-----|------|
| ⭐ | MRDB (Mamba+ResNet, 2024) | **0.9002** | 0.8911 | — | — | PMC/Bioengineering |
| ⭐ | ResUNet (2025) | 0.8424 | 0.8898 | — | — | AIMS Medical Science |
| ⭐ | DPAM-UNet++ (2024) | 0.8310 | 0.8702 | — | 0.9213 | BMC Medical Imaging |
| ⭐ | TRFE-Net (2021) | ~0.830 | — | — | — | ISBI'21 |
| ⭐ | U-Net baseline | ~0.78~0.82 | — | — | — | 복수 논문 |
| — | — | — | — | — | — | — |
| 🥇 1 | `ce_dice` | **0.8829** | 0.8977 | 0.9851 | **0.9939** | 기준선 |
| 🥈 2 | `lwce_dice` | 0.8779 | 0.8929 | 0.9844 | 0.9934 | |
| 🥉 3 | `plwce_dice` (α=6.90) | 0.8775 | 0.8868 | **0.9852** | 0.9936 | |
| 4 | `cb_dice` | 0.8755 | **0.8986** | 0.9830 | 0.9937 | |
| 5 | `plwce_focal_dice` (α=9.26, γ=1.88) | 0.8646 | 0.9164 | 0.9776 | 0.9933 | |
| 6 | `wce_dice` | 0.8468 | 0.9436 | 0.9686 | 0.9935 | |

> **비고**: 최우수(ce_dice 0.8829)가 DPAM-UNet++(0.831), TRFE-Net(0.830) 상회. MRDB(0.900) 대비 -0.017.
> `pwce_dice` 미실험.

---

### 1-6. LiTS 2017 — Liver Tumor Segmentation (3-class)

| 모달리티 | 태스크 | 불균형 | 평가셋 | Train/Val 슬라이스 |
|---------|--------|--------|--------|-----------------|
| 복부 CT (HU -100~250) | 3-class (BG/Liver/Tumor) | BG:Liver **15:1**, BG:Tumor **356:1** | val | 5433 / 1369 |

| 순위 | 방법 | mDice ↓ | Liver Dice | Tumor Dice | 비고 |
|-----|------|--------|-----------|-----------|------|
| ⭐ | nnU-Net (2021) | — | **~0.9894** | ~0.70~0.75 | Nature Methods |
| ⭐ | ASLseg (2024) | — | — | **0.7428** | Medical Image Analysis |
| ⭐ | Swin-UNet (2021) | — | ~0.956 | ~0.681 | ECCV'22 |
| ⭐ | U-Net++ baseline | — | ~0.950 | ~0.60~0.65 | 복수 논문 |
| — | — | — | — | — | — |
| 🥇 1 | `ce_dice` | **0.5342** | **0.9410** | 0.1274 | 기준선 |
| 🥈 2 | `plwce_dice` (α=4.40) | 0.5327 | 0.9248 | **0.1406** | |
| 🥉 3 | `wce_dice` | 0.5313 | 0.9235 | 0.1390 | |
| 4 | `cb_dice` | 0.5287 | 0.9254 | 0.1319 | |
| 5 | `lwce_dice` | 0.5265 | 0.9315 | 0.1214 | |

> **비고**: Liver Dice는 CE(0.941)가 최고이나 SoTA(0.989) 대비 크게 낮음.
> Tumor Dice는 전 방법 0.12~0.14로 SoTA(0.74)에 훨씬 못 미침 — 극단적 불균형(356:1)의 한계.
> `plwce_focal_dice`, `pwce_dice` 미실험.

---

### 1-7. Pancreas/Synapse — Multi-organ CT Segmentation (8-class)

| 모달리티 | 태스크 | 불균형 | 평가셋 | 모델 |
|---------|--------|--------|--------|------|
| 복부 CT | 8-class (대동맥·담낭·비장 등) | 기관별 상이 | test | TransUNet |

| 순위 | 방법 | mDice ↓ | Aorta | GB | Spleen | L.K | R.K | Liver | Stomach | Pancreas | 비고 |
|-----|------|--------|------|-----|--------|-----|-----|-------|---------|---------|------|
| ⭐ | DS-UNETR++ (2025) | **0.8775** | — | — | — | — | — | — | — | — | arXiv |
| ⭐ | DIN (2025) | 0.8549 | — | — | — | — | — | — | — | — | Medical Image Analysis |
| ⭐ | SwinUNet (2021) | 0.7913 | — | — | — | — | — | — | — | — | ECCV'22 |
| ⭐ | TransUNet 원본 (2021) | 0.7748 | — | — | — | — | — | — | — | — | arXiv |
| ⭐ | U-Net baseline | ~0.68~0.74 | — | — | — | — | — | — | — | — | 복수 논문 |
| — | — | — | — | — | — | — | — | — | — | — | — |
| 🥇 1 | `plwce_dice` (α=6.44) | **0.7619** | 0.826 | 0.469 | **0.808** | 0.766 | 0.937 | 0.577 | **0.891** | **0.822** | |
| 🥈 2 | `ce_dice` | 0.7589 | 0.826 | **0.485** | 0.793 | **0.770** | 0.940 | **0.586** | 0.883 | 0.788 | 기준선 |
| 🥉 3 | `plwce_focal_dice` (α=8.37, γ=4.76) | 0.7426 | 0.811 | 0.466 | 0.777 | 0.745 | **0.941** | 0.558 | 0.856 | 0.789 | |
| 4 | `pwce_dice` (α=3.98) | 0.6510 | 0.808 | 0.001 | 0.777 | 0.718 | 0.908 | 0.500 | 0.816 | 0.680 | GB 붕괴 |

> **비고**: plwce_dice가 TransUNet 원본(0.7748)에 근접(0.7619). SwinUNet(0.791) 대비 -0.029.
> pwce_dice는 담낭(GB) Dice 0.001로 완전 붕괴. `lwce_dice`, `wce_dice`, `cb_dice` 미실험.

---

### 1-8. WMH 2017 — Brain White Matter Hyperintensity Segmentation

| 모달리티 | 태스크 | 불균형 | 평가셋 | 입력 채널 |
|---------|--------|--------|--------|---------|
| 뇌 MRI (FLAIR+T1) | Binary (BG/WMH) | **250:1** | val | 3ch (FLAIR, T1, FLAIR) |

| 순위 | 방법 | Dice ↓ | Sensitivity | Specificity | AUC | 비고 |
|-----|------|--------|------------|------------|-----|------|
| ⭐ | nnU-Net 3D (2021) | ~0.800 | — | — | — | Nature Methods (Utrecht 단일 사이트) |
| ⭐ | Robust-WMH-UNet (2026) | 0.768 | — | — | — | arXiv |
| ⭐ | Transformer-based (2025) | 0.720 | — | — | — | NeuroImage |
| ⭐ | 2D U-Net | ~0.75~0.79 | — | — | — | WMH Challenge |
| — | — | — | — | — | — | — |
| 🥇 1 | `plwce_dice` (α=3.27) | **0.8862** | 0.8879 | 0.9994 | 0.9979 | |
| 🥈 2 | `cb_dice` | 0.8814 | 0.8763 | **0.9995** | 0.9874 | |
| 🥉 3 | `lwce_dice` | 0.8812 | 0.8777 | 0.9994 | 0.9956 | |
| 4 | `plwce_focal_dice` (α=11.29, γ=4.88) | 0.8809 | 0.8819 | 0.9994 | 0.9975 | |
| 5 | `ce_dice` | 0.8759 | 0.8739 | 0.9994 | 0.9919 | 기준선 |
| 6 | `wce_dice` | 0.8681 | **0.9565** | 0.9988 | **0.9997** | |

> **비고**: 전 방법이 nnU-Net 3D(0.800), 2D U-Net(0.79) SoTA 상회.
> 높은 불균형(250:1)에서 plwce_dice가 CE 대비 +0.010 향상.
> wce_dice는 Dice 최저이나 AUC·Sensitivity 최고 — FG 탐지 편향.
> `pwce_dice` 미실험.

---

## 2. 불균형 비율별 손실함수 효과 분석

| 불균형 비율 | 도메인 | 🥇 최우수 손실함수 | CE 대비 Dice 변화 | 특이사항 |
|-----------|--------|----------------|----------------|---------|
| **~2.6:1** | MoNuSeg (Nuclei) | `ce_dice` | ±0.000 (기준선 = 최우수) | 손실함수 간 차이 최대 0.017 |
| **~3.6:1** | ISIC 2018 (Skin) | `lwce_dice` | +0.006 | 낮은 불균형에서도 lwce 소폭 효과 |
| **~5~10:1** | Kvasir-SEG (Polyp) | `lwce_dice` | +0.005 | Focal 계열 이점 없음 |
| **~6:1** | TN3K (Thyroid) | `ce_dice` | ±0.000 (기준선 = 최우수) | 초음파 노이즈·경계 불명확 도메인 |
| **~9:1** | DRIVE (Retinal) | `plwce_focal_dice` | +0.005 | Sensitivity/Specificity 균형 개선 |
| **15:1** | LiTS (Liver) | `ce_dice` | ±0.000 (기준선 = 최우수) | Liver는 CE 최우수 |
| **356:1** | LiTS (Tumor) | `plwce_dice` | +0.013 (Tumor Dice) | 전체 방법 Tumor Dice 0.12~0.14로 저조 |
| **다중** | Pancreas (8-class) | `plwce_dice` | +0.003 (mDice) | 소수 클래스(pancreas) +0.034 |
| **~250:1** | WMH (Brain) | `plwce_dice` | +0.010 | 높은 불균형에서 PLWCE 효과 뚜렷 |

### 소견

**불균형이 낮은 경우 (≤5:1)**
- MoNuSeg(2.6:1): 손실함수 선택의 영향이 제한적. CE 기준선이 최우수 또는 동등
- ISIC(3.6:1), Kvasir(~5:1): `lwce_dice`가 소폭 우세하나 차이 미미

**불균형이 중간인 경우 (5~20:1)**
- TN3K(6:1): CE 기준선 최우수. 초음파 도메인 특성(speckle noise, 불명확한 경계)이 손실함수 효과보다 지배적일 수 있음
- DRIVE(~9:1): `plwce_focal_dice` 우세 — Dice와 Sensitivity 균형 측면에서 이점

**불균형이 높은 경우 (>50:1)**
- WMH(250:1): `plwce_dice`가 가장 일관적으로 우수 (+0.010). PLWCE의 power-law 가중치가 극단적 불균형에 효과적
- LiTS Tumor(356:1): 극단적 불균형에서 손실함수 개선 한계. 아키텍처·전처리 개선이 더 중요할 것으로 판단

**Multi-class**
- Pancreas: `plwce_dice`가 소수 클래스(pancreas, spleen) 개선에 효과적
- `pwce_dice`는 Pancreas에서 담낭(GB) Dice 0.001 붕괴 — 불안정성 주의

---

## 3. SoTA 대비 요약

| 도메인 | 본 연구 최고 Dice | 비교 SoTA 방법 | SoTA Dice | 격차 | 모델 차이 |
|--------|----------------|--------------|----------|------|---------|
| Kvasir-SEG | 0.899 (lwce) | PraNet (2021) | 0.898 | **≈동등** | 동급 (U-Net계열) |
| DRIVE | 0.696 (plwce_f) | IterNet 원본 | ~0.779 | -0.083 | 평가셋 차이 (val 4장) |
| ISIC 2018 | 0.902 (lwce) | UNet++ (2019) | ~0.890 | **+0.012** | 동급 |
| MoNuSeg | 0.817 (ce) | HoverNet | 0.826 | -0.009 | 전용 모델 vs U-Net |
| TN3K | 0.883 (ce) | DPAM-UNet++ | 0.831 | **+0.052** | 동급 대비 우세 |
| LiTS Liver | 0.941 (ce) | U-Net++ baseline | ~0.950 | -0.009 | 동급 |
| LiTS Tumor | 0.141 (plwce) | U-Net++ baseline | ~0.600 | -0.459 | 극단적 불균형 한계 |
| Pancreas | 0.762 (plwce) | TransUNet 원본 | 0.775 | -0.013 | 동일 아키텍처 |
| WMH | 0.886 (plwce) | 2D U-Net | ~0.790 | **+0.096** | 동급 대비 크게 우세 |

---

## 4. 미완료 도메인 (업데이트 예정)

| 도메인 | 예상 불균형 | 예상 최우수 손실함수 | 비고 |
|--------|-----------|----------------|------|
| BUSI (Breast US) | ~4:1 (추정) | `lwce_dice` or `ce_dice` | 진행 중 |
| ACDC (Cardiac MRI) | BG:구조 ~9:1 | `plwce_dice` or `lwce_dice` | 진행 중 |
| REFUGE (Optic Disc/Cup) | Disc:Cup ~4:1 | 미정 | 진행 중 |

---

## 5. 업데이트 가이드

새 실험 완료 후 추가할 위치:
1. **섹션 1**: 새 도메인 섹션 추가 (SoTA ⭐행 + 본 연구 결과 행)
2. **섹션 2**: 불균형 비율별 표에 행 추가 + 소견 업데이트
3. **섹션 3**: SoTA 대비 요약 표에 행 추가
4. **섹션 4**: 해당 행 삭제

```markdown
### 1-X. {도메인명}

| 모달리티 | 태스크 | 불균형 | 평가셋 | 기타 |

| 순위 | 방법 | Dice ↓ | Sensitivity | Specificity | AUC | 비고 |
|-----|------|--------|------------|------------|-----|------|
| ⭐  | {SoTA 방법} | {값} | ... | ... | ... | 출처 |
| —  | — | — | — | — | — | — |
| 🥇 1 | `{손실함수}` | {값} | ... | ... | ... | |
| ... | ... | ... | ... | ... | ... | |
```
