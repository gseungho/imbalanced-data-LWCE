# Boundary Loss Ablation — 실험 설계 및 배경

## 실험 동기

기존 실험(`medical_data/results/`)에서 **LWCE/PLWCE + Dice ≈ CE + Dice** 수준의 결과가 반복 관찰됨.

**원인 분석**:
- Dice Loss는 `2·TP / (예측 병변 + 실제 병변)` 형태로 **분모 정규화를 통해 소수 클래스 무시를 이미 방지**함
- 즉, Dice가 "병변을 예측하지 않으면 loss 최대" 패턴을 구현 → LWCE의 역할과 중복
- CE 항(전체 loss의 50%)에 LWCE 가중치를 줘도, Dice가 이미 해당 실패 모드를 막고 있으므로 효과 희석

**결론**: Dice와 LWCE는 **같은 문제(병변 무시 방지)**를 다른 방식으로 해결 → 중복

---

## 새로운 실험 방향

**Boundary Loss (BL)** 은 Dice와 완전히 다른 기하학적 속성을 최적화:

| | Dice | Boundary Loss |
|---|---|---|
| 최적화 대상 | 영역 겹침 (region overlap) | 경계선 거리 (boundary distance) |
| 소수 클래스 무시 방지 | ✅ (분모 정규화) | ❌ (위치만 봄) |
| 경계 정밀도 | ❌ | ✅ |

→ **CE + Dice + BL** 이 문헌(Kervadec et al., 2019)에서의 표준 조합

→ 여기서 CE를 **LWCE/PLWCE** 로 교체하면 Dice와 중복 없는 방식으로 LWCE를 적용 가능

---

## Log-Boundary Loss (LBL) — 신규 제안

표준 BL의 거리 맵 `φ_G(x)`:
- 이미지 크기에 의해 상한이 존재하나 (256px → 최대 ~180), 배경 중심부 픽셀이 경계 근처보다 지나치게 큰 가중치를 받음
- "명백한 배경" 픽셀(쉬운 샘플)에 과도한 집중

**LBL**: `log(1 + φ_G(x))` 적용 → 거리 범위를 압축, 쉬운 픽셀 비중 감소

```
BL  weight ∝ |d|           → 선형 (배경 중심부 과대)
LBL weight ∝ log(1+|d|)    → 로그 압축 (Focal Loss와 유사한 효과)
```

LWCE의 핵심 철학("log scaling으로 가중치 폭발 억제")을 픽셀 거리 도메인에 확장.

**주의**: BL과 달리 LBL은 "폭발 방지"가 아니라 "쉬운 픽셀 비중 감소"가 핵심 차이점.

---

## 실험 조합 (5종) — LBL 제외 확정

| # | loss_name | 설명 | Dice | BL | PLWCE |
|---|---|---|---|---|---|
| 1 | `ce_dice` | 기존 베이스라인 | ✅ | ❌ | ❌ |
| 2 | `plwce_dice` | 기존 PLWCE 베스트 | ✅ | ❌ | ✅ |
| 3 | `ce_dice_boundary` | 문헌 베이스라인 (Kervadec 2019) | ✅ | BL | ❌ |
| 4 | `plwce_dice_boundary` | **핵심 실험** | ✅ | BL | ✅ |
| 5 | `plwce_boundary` | Dice 제거 실험 (성능 하락 예상) | ❌ | BL | ✅ |

**LBL(Log-Boundary Loss) 제외 이유**: 3개 도메인 실험 결과 BL 대비 유의미한 성능 차이 없음 → 실험군에서 제거.

**핵심 비교**: `plwce_dice_boundary` vs `ce_dice_boundary` — PLWCE가 BL 환경에서 추가 이점을 갖는가?

---

## Optuna 전략

기존 실험 alpha와 BL 환경 최적 alpha가 다를 수 있으므로 **별도 탐색**:

- `plwce_dice_boundary` → `best_alpha_with_dice` (plwce_dice_log_boundary에도 재사용)
- `plwce_boundary` → `best_alpha_without_dice` (plwce_log_boundary에도 재사용)
- proxy: subset_ratio=0.15, epochs=8, n_trials=20
- 탐색 방식: **GridSampler** (TPE/MedianPruner 미사용) — 균일 20등분으로 전체 범위 커버 (탐색 순서는 랜덤)

기존 실험 최적 alpha (fallback):
- WMH: 3.275
- ISIC 2018: 12.721
- TN3K: 6.896

---

## 도메인별 설정

| 도메인 | 불균형비 | 기존 1위 | FINAL_EPOCHS | Early Stop | optimizer | 모델 |
|---|---|---|---|---|---|---|
| WMH | ~250:1 (극심) | plwce_dice | 75 | patience=15 | Adam | U-Net (ResNet34) |
| ISIC 2018 | ~4:1 (중간) | lwce_dice | 75 | patience=15 | Adam | U-Net (ResNet34) |
| TN3K | ~6:1 (중간) | ce_dice | 75 | patience=15 | AdamW | U-Net (ResNet34) |
| DRIVE | ~9:1 (중간) | plwce_focal_dice | 75 | patience=15 | Adam | IterNet |
| LiTS | Tumor ~356:1 (극심) | plwce_dice | 75 | patience=15 | Adam+CosineAnneal | U-Net++ (ResNet50) |
| Kvasir | ~5:1 (중간) | lwce_dice | 75 | patience=15 | Adam | PraNet (Res2Net50) |

> DRIVE: test GT 없음 → val set으로 평가  
> LiTS: 3-class (BG/Liver/Tumor), 평가 지표 = Liver_Dice / Tumor_Dice / mDice  
> Kvasir: PraNet 4-출력 구조, 각 출력에 손실 합산

---

## 결과 저장 위치

```
boundary_ablation/results/
├── wmh/
│   ├── wmh_boundary_optuna.json       # Optuna 결과
│   ├── wmh_boundary_ablation.json     # 최종 평가 결과 (JSON)
│   ├── wmh_boundary_ablation.xlsx     # 최종 평가 결과 (Excel: Summary + Training_History 시트)
│   ├── boundary_ablation_curves.png   # 학습 곡선
│   └── boundary_ablation_bar.png      # Test Dice 바차트
├── isic2018/
├── tn3k/
├── drive/                             # (신규) 망막 혈관, val set 평가
├── lits/                              # (신규) 간/종양, Liver_Dice+Tumor_Dice+mDice
└── kvasir/                            # (신규) 폴립, test set 평가
```

---

## Lambda Annealing (구현 완료)

BL은 초반 학습이 불안정하므로 **지연 선형(delayed linear)** annealing 적용.

**공식**:
```
WARMUP = int(epochs × 0.2)   # 처음 20% 에포크: BL = 0 고정

epoch ≤ WARMUP  →  α_t = 0
epoch > WARMUP  →  α_t = min((epoch - WARMUP) / (epochs - WARMUP), 1.0) × alpha_max
```

**alpha_max (컴포넌트 수에 따라 분리)**:
| 조합 | alpha_max | 최종 가중치 |
|---|---|---|
| 3-comp (CE+Dice+BL) | **1/3** | CE = Dice = BL = 0.333 (균등) |
| 2-comp (CE+BL) | **0.5** | CE = BL = 0.5 (균등) |

**코드 (train_model 내 epoch 루프)**:
```python
if criterion.boundary_loss is not None:
    WARMUP = int(epochs * 0.2)
    has_region = criterion.region_loss is not None
    alpha_max = 1/3 if has_region else 0.5
    if epoch <= WARMUP:
        alpha_t = 0.0
    else:
        alpha_t = min((epoch - WARMUP) / (epochs - WARMUP), 1.0) * alpha_max
    criterion.set_boundary_alpha(alpha_t)
```

**변경 이유**:
- 기존 `min(epoch/epochs, 0.5)`: 3-comp에서 BL 최종 50% → CE·Dice 각 25%로 BL 편향 심함
- DRIVE CE+Dice+BL: Dice 0.530으로 급락 (baseline 0.611) — 과도한 BL 비중이 원인으로 판단
- 워밍업 없이 epoch 1부터 BL 증가 → 초반 불안정 학습 노출

**Early Stopping**: patience=15 (모든 도메인 공통, 75 epoch 기준)

---

## 참고 문헌

- Kervadec et al. (2019). "Boundary loss for highly unbalanced segmentation." *MIDL 2019*.
- Abraham & Khan (2019). "A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation." *ISBI 2019*.
