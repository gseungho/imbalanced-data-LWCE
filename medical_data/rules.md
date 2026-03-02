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
| `cb` | w = (1-β) / (1-β^n) | Effective Number 기반 |

- 모든 가중치는 **평균으로 정규화** (`weights / mean(weights)`)
- `alpha` 범위: 실험상 2.5~15.0 (Optuna로 탐색)
- `beta` 기본값: 0.9999 (CB Loss용)

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

### 3-2. Retinal Vessel Image (DRIVE)
- **모델**: IterNet (경량 CNN) — PraNet도 비교 가능
- **태스크**: Binary segmentation (혈관 vs 배경)
- **데이터**: DRIVE 20장 (train/val split), test에 GT 없음
- **불균형**: BG:FG = **10.7:1** (심각)
- **전처리**: CLAHE (Green 채널 강조), 256×256 Patch 기반 샘플링
- **실험 Loss**: `ce_dice`, `wce_dice`, `lwce_dice`, `plwce_dice`
- **결과**: `lwce_dice` 최고 (Dice=0.7033)
- **평가 지표**: Dice, Sensitivity, Specificity, AUC

### 3-3. Pancreas / Multi-Organ CT (Synapse)
- **모델**: TransUNet (R50+ViT-B/16)
- **태스크**: 9-class segmentation (배경 + 8개 장기)
- **데이터**: Synapse Multi-organ CT (.npz 슬라이스 / .h5 볼륨)
- **불균형**: 췌장 등 소수 장기 극심한 불균형
- **하이퍼파라미터 탐색**: Optuna (alpha 범위 2.5~15.0, 30 trials, subset_ratio=0.15)
- **실험 Loss**: `ce_dice`, `plwce_dice`, `pwce_dice`
- **평가 지표**: 클래스별 Dice + mDice (background 제외)

---

## 4. 추가 연구 계획 (신규 도메인)

### 4-0. 완료된 신규 도메인

| 도메인 | 파일 | 모델 | 상태 |
|--------|------|------|------|
| 피부 병변 (ISIC 2018) | `Skin_Lesion_ISIC2018.ipynb` | U-Net (ResNet34) | 코드 완성, 실행 대기 |



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
- **Val 비율**: 0.1~0.2 (데이터셋 크기에 따라 조정)
- **Test set**: 가능하면 공식 test split 사용; 없으면 val set으로 평가하되 명시

### 5-3. 클래스 비율 계산 규칙
- 학습 데이터 전체에 대해 픽셀 단위로 계산
- `class_counts` 리스트로 저장 후 `get_loss_function()`에 전달
- 비율 출력 필수: `print(f"BG:FG = {ratio:.1f}:1")`

### 5-4. 모델 저장 규칙
- Best Val Dice 기준으로만 저장
- 파일명: `/tmp/best_{model_name}_{loss_name}.pth`
- 최종 실험 결과는 JSON으로 저장: `/tmp/{domain}_final_results.json`

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
# proxy 설정: subset_ratio=0.15, epochs=5, n_trials=30
```

### 6-5. 시각화 규칙
- `matplotlib.use('Agg')` 서버 환경에서 필수
- 학습 곡선 (Loss + Val Metric) 반드시 저장
- 예측 시각화: Input / GT / Prob Map / Pred 4열 구성

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
| Loss Function | alpha | Dice | Sens | Spec | AUC |
|---------------|-------|------|------|------|-----|
| CE+Dice       | -     |      |      |      |     |
| WCE+Dice      | -     |      |      |      |     |
| LWCE+Dice     | -     |      |      |      |     |
| PLWCE+Dice    | best  |      |      |      |     |
| CB+Dice       | -     |      |      |      |     |
```

- alpha는 Optuna 최적값 기재
- 최고 성능 수치 **볼드** 처리
- 불균형 비율(BG:FG)도 함께 기재

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
