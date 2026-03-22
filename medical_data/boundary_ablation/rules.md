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

## 실험 조합 (8종)

| # | loss_name | 설명 | Dice | BL | PLWCE |
|---|---|---|---|---|---|
| 1 | `ce_dice` | 기존 베이스라인 | ✅ | ❌ | ❌ |
| 2 | `plwce_dice` | 기존 PLWCE 베스트 | ✅ | ❌ | ✅ |
| 3 | `ce_dice_boundary` | 문헌 베이스라인 | ✅ | BL | ❌ |
| 4 | `plwce_dice_boundary` | **핵심 실험** | ✅ | BL | ✅ |
| 5 | `plwce_dice_log_boundary` | **핵심 실험** | ✅ | LBL | ✅ |
| 6 | `ce_dice_log_boundary` | Dice+LBL 효과 분리 | ✅ | LBL | ❌ |
| 7 | `plwce_boundary` | Dice 제거 실험 | ❌ | BL | ✅ |
| 8 | `plwce_log_boundary` | Dice 제거 실험 | ❌ | LBL | ✅ |

**7, 8번 예상**: 성능 하락 가능성 높음 (BL만으로 region coverage 보장 불가)

**핵심 비교**: `plwce_dice_boundary` vs `ce_dice_boundary` — PLWCE가 BL 환경에서 추가 이점을 갖는가?

---

## Optuna 전략

기존 실험 alpha와 BL 환경 최적 alpha가 다를 수 있으므로 **별도 탐색**:

- `plwce_dice_boundary` → `best_alpha_with_dice` (plwce_dice_log_boundary에도 재사용)
- `plwce_boundary` → `best_alpha_without_dice` (plwce_log_boundary에도 재사용)
- proxy: subset_ratio=0.15, epochs=5, n_trials=20

기존 실험 최적 alpha (fallback):
- WMH: 3.275
- ISIC 2018: 12.721
- TN3K: 6.896

---

## 도메인별 설정

| 도메인 | 불균형비 | 기존 1위 | FINAL_EPOCHS | optimizer |
|---|---|---|---|---|
| WMH | ~100:1 (극심) | plwce_dice | 50 | Adam |
| ISIC 2018 | ~4:1 (중간) | lwce_dice | 30 | Adam |
| TN3K | ~3:1 (낮음) | ce_dice | 50 | AdamW |

---

## 결과 저장 위치

```
boundary_ablation/results/
├── wmh/
│   ├── wmh_boundary_optuna.json      # Optuna 결과
│   ├── wmh_boundary_ablation.json    # 최종 평가 결과
│   ├── boundary_ablation_curves.png
│   └── boundary_ablation_bar.png
├── isic2018/
└── tn3k/
```

---

## 참고 문헌

- Kervadec et al. (2019). "Boundary loss for highly unbalanced segmentation." *MIDL 2019*.
- Abraham & Khan (2019). "A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation." *ISBI 2019*.
