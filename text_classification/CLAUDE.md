# text_classification — GoEmotions Long-Tail 텍스트 분류 실험

## 프로젝트 개요

**목적**: 논문의 **아키텍처 독립성(architecture-independence)** 주장을 검증. `image_classification`(CNN, ResNet-32) 실험을 **텍스트 모달리티 + 트랜스포머 백본**으로 확장해, 동일한 LWCE 계열 손실이 백본·모달리티를 바꿔도 작동함을 보인다.

- **데이터**: GoEmotions (`simplified` config) → **single-label 필터** → 28-class(27 감정 + neutral) 단일 라벨 분류. 자연 long-tail 불균형.
- **백본**: `transformer_text.build_text_transformer` — **사전학습 없음**(from-scratch). ViT/ResNet from-scratch 결정과 일관.
- **손실**: `custom_losses.get_clf_loss` — `image_classification/custom_losses.py`와 **동일 파일 복사본**(backbone-agnostic classification loss).
- **프로토콜**: 자연 불균형 단일 설정(**IR 루프 없음** — CIFAR와 다른 점), 5 seeds, F1-Macro 주지표.

---

## 폴더 구조

```
text_classification/
├── custom_losses.py       # 분류용 손실 11종 (image_classification와 동일 파일 — 수정 시 양쪽 동기화)
├── transformer_text.py    # 소형 트랜스포머 인코더 (from-scratch)
├── GoEmotions_LT.ipynb    # 실험 노트북 (셀 0~8; Cell 8 = 3차 피드백 분석)
├── CLAUDE.md              # 이 파일
└── results/GoEmotions/
    ├── optuna_checkpoint.json / optuna_trials.json / results_checkpoint_v2.json
    ├── results_agg.json / results_per_seed.json / results.xlsx / analysis_summary.json
    └── *.png (class_distribution / training_curves / f1_macro / group_accuracy /
               optimization_stability / class_weight_distribution / sensitivity_1d /
               sensitivity_combined_2d)
```

---

## 백본 — transformer_text.py

소형 트랜스포머 인코더. 입력 `(B, L)` 토큰 id.

```
Embedding(vocab, d_model, padding_idx=0) + 학습형 PositionalEmbedding(max_len)
    ↓
N × TransformerEncoderLayer (d_model=128, nhead=4, layers=4, ff=256, GELU, batch_first)
    ↓  (src_key_padding_mask = pad 위치)
masked mean-pool (비-pad 토큰 평균) → LayerNorm → Dropout → Linear(d_model, num_classes)
```

- 팩토리: `build_text_transformer(vocab_size, num_classes, pad_idx=0, max_len=64)`
- 파라미터 ~0.6M (vocab 500 기준, 실제 vocab에 비례).
- pad_idx=0, `<unk>`=1 로 vocab 구성.

---

## 손실함수 (11종) — image_classification와 동일 파일

`ce, wce, pwce, sqce, lwce, plwce, eslwce, combined, cb, focal, logitadj`.
**lwce/plwce/eslwce가 proposed**, `combined`는 ablation 전용.

| loss | 공식 | 파라미터 | 역할 |
|------|------|---------|------|
| `eslwce` | `1/(log1p(n)+ε)` | eps | **proposed** — 3차 §3.4 |
| `combined` | `1/(log1p(n)+ε)^α` | alpha, eps | §4.5 ablation 마지막 행 (orthogonality) |
| `logitadj` | `CE(logits + τ·log prior)` | tau | §4.2 baseline (LDAM 대안) |

> `combined`는 **α=1 → ES-LWCE**, **ε→0 → PLWCE**로 정확히 환원됨(테스트로 확인).
> 이 환원 성질이 orthogonality 주장의 전제라 바꾸면 안 됨.

**파싱 우선순위**: `combined > plwce > eslwce > lwce > pwce > sqce > wce > cb > ce`
> `'eslwce'`가 `'lwce'`를 부분문자열로 포함 → **반드시 `lwce`보다 먼저** 검사.
> `logitadj`는 가중치 모드가 아니라 criterion 교체 (weight_mode='ce'로 떨어짐).

### ES-LWCE (ε) 성질 — 튜닝 전 필독

`w_c = 1 / (log(1+n_c) + ε)`. 가중치는 `calculate_weights()` 끝에서 **평균 정규화**되므로
ε의 절대 스케일은 무의미하고 **가중치 분포를 얼마나 평평하게 만드는지만** 남는다.

압축비 `(d_freq+ε)/(d_rare+ε)`를 ε로 미분하면 `(d_rare−d_freq)/(d_rare+ε)² < 0`
→ **ε에 대해 단조 감소**. 즉 **ε는 순수한 "완화" 다이얼**:

```
ε → 0 : LWCE        ε → ∞ : CE(균일)
```

**ES-LWCE는 절대 LWCE보다 공격적일 수 없다.** GoEmotions 실측(28-class, min=39, max=12823):

| ε | 0.01 | 0.1 | 0.5 | 1 | 3 | 10 | 100 |
|---|---|---|---|---|---|---|---|
| LWCE 대비 압축비 | 99.8% | 98.4% | 92.7% | 87.0% | 72.6% | 55.4% | 41.2% |

> **예상 결과**: GoEmotions의 F1 최적 α\*=3.97 (LWCE의 α=1보다 **공격적** 쪽).
> ε는 반대 방향(완화)으로만 갈 수 있으므로 **ε\*는 그리드 하한(0.1)에 붙고
> ES-LWCE ≈ LWCE로 수렴**할 가능성이 높다. 이는 실패가 아니라 **"ε는 정확도 knob이
> 아니라 n_c=0 대응 + 상한 안전장치"라는 논문 주장(교수님 2차 피드백 M2)의 실증**이다.

---

## 데이터 (GoEmotions single-label)

- `load_dataset('google-research-datasets/go_emotions', 'simplified')` — 공식 train/val/test split 사용.
  > ⚠️ 최신 `datasets`는 네임스페이스 포함 정식 ID를 요구함. bare `'go_emotions'`는 **HfUriError** 발생.
- **single-label 필터**: `len(ex['labels']) == 1` 인 예시만 사용 (multi-label → BCE 전환은 CE 기반 LWCE와 불일치하므로 제외).
- 토크나이저: 정규식 `[a-z0-9']+` word-level, train에서 vocab 구축(`MIN_FREQ=2`), `MAX_LEN=64` 패딩/절단.
- `class_counts`는 train 라벨 분포에서 계산. 클래스명: `features['labels'].feature.names`.

---

## 학습 설정 (트랜스포머용)

```python
BATCH_SIZE=64, MAX_LEN=64, NUM_WORKERS=0, FINAL_EPOCHS=50, WARMUP_EPOCHS=3
optimizer = AdamW(lr=5e-4, weight_decay=0.01)
scheduler = LambdaLR (linear warmup 3ep → cosine decay)
grad clip = 1.0   # 트랜스포머 안정화
```

> **CIFAR와 차이**: ResNet은 SGD lr=0.1 + MultiStepLR. 트랜스포머 from-scratch는 SGD로 안 되어 AdamW+warmup+cosine+grad-clip 필수. **데이터·손실·평가·저장은 CIFAR와 동일하게 미러링**, 옵티마이저와 IR 루프 제거만 다름.

---

## Optuna / 평가 / 저장

- **목적함수 = F1-Macro** (`compute_val_acc()`가 리턴하는 값). balanced_acc 아님 — 주의.
  - ⚠️ `train_model`의 history 키 `val_balanced_acc`는 **레거시 이름이고 실제 값은 F1-Macro**.
    기존 체크포인트 호환 때문에 키 이름은 유지함.
- GridSampler 20 trials × 6 study, stratified proxy(min 5/class, ratio 0.20, 10 epochs):

  | loss | 파라미터 | 범위 | 스케일 |
  |------|---------|------|--------|
  | pwce | alpha | [0.3, 5.0] | linspace |
  | plwce | alpha | [0.5, 6.0] | linspace |
  | focal | gamma | [1.0, 5.0] | linspace (γ<1 NaN 붕괴) |
  | eslwce | eps | [0.1, 10.0] | **logspace** (효과가 배수적) |
  | logitadj | tau | [0.25, 2.0] | linspace |
  | combined | alpha × eps | [0.5,6.0] × [0.1,10] | **2D 5×4 = 20** |

- **trial 전체 저장** → `optuna_trials.json` (sensitivity 곡선용, 추가 학습 없음).

- **증분 실행**: Cell 5는 `CKPT_OPTUNA`를 로드한 뒤 **빠진 study만** 돌린다(`pending` 리스트).
  체크포인트가 있으면 통째로 skip하던 기존 구조는 **새 study가 영원히 안 돌아가는 버그**여서 교체함.
- **단일 설정이라 IR 루프 없음** → `optuna_best`는 flat dict (`{'plwce': {'alpha': ...}, 'eslwce': {'eps': ...}}`).
- 평가: `compute_val_metrics`(F1-Macro/Balanced/Top1 + Many/Medium/Few tertile split). CIFAR와 동일 로직(logits→confusion matrix).
- 저장: `results/GoEmotions/` 아래 JSON/xlsx/PNG + run별 체크포인트(`{loss}_s{seed}`).

---

## 3차 피드백 (2026-07-12) 대응 — 노트북 반영 완료

**핵심 원칙**: "학습이 끝나면 복구 불가능한 값"만 루프 안에서 잡는다.

| 요구 (3차 §) | 복구 가능? | 처리 |
|---|---|---|
| Gradient Norm (§4.7) | ❌ | `clip_grad_norm_` 리턴값 = 클리핑 전 total norm → `history['grad_norm']` |
| Min/Maj Gradient Ratio (§4.7) | ❌ | `logits.retain_grad()` → 샘플별 로짓 grad norm을 Many/Few로 집계 |
| Training Loss (§4.7) | ✅ | 이미 있음 |
| Class Weight Distribution (§4.7) | ✅ | `class_counts`로 언제든 재계산 (`get_weights()`) |
| G-Mean / Worst-class / Minority Recall (§4.3) | ✅ | 전부 `Per_Class_Acc`에서 사후 계산 |
| Friedman / Wilcoxon (§4.8) | ✅ | 저장된 결과로 사후 처리 (Cell 8) |
| Sensitivity (§4.6) | ✅ | `study.trials` 전체 저장 → 추가 학습 0 |

> **⚠️ Gradient 계측 추가로 v1(40런)은 폐기됨** → `CKPT_RESULTS = results_checkpoint_v2.json`.
> Optuna 체크포인트(`optuna_checkpoint.json`)는 proxy 프로토콜이 안 바뀌어 **재사용**(pwce/plwce/focal).

### Weight Normalization (3차 §3.3) — 이미 충족

교수님이 명시를 요구한 `w̃_c = C·w_c / Σⱼwⱼ`가 `calculate_weights()`의
`weights / np.mean(weights)`와 **수학적으로 동일**(`mean = Σw/C`). 게다가 **모든 모드에
무조건 적용**되므로 WCE·CB도 동일 scale convention → "공정 비교" 요구도 충족.
**논문에 문장만 추가하면 됨. 코드 수정 불필요.**

### Proposition 3 실증 (GoEmotions IR=328.8) — 논문 표 후보

| Loss | w_rare/w_freq | 이론 |
|------|--------------|------|
| wce | **328.79** | ρ (선형) — IR과 정확히 일치 |
| pwce (α=2) | 108,106 | ρ² |
| sqce | 18.13 | √ρ |
| **lwce** | **2.56** | O(log ρ) — **128배 압축** |
| plwce (α=2) | 6.58 | (log ρ)² |
| cb | 185.58 | — |

### 통계 검정 주의

Cell 8의 Friedman/Wilcoxon은 **seed를 블록**으로 씀 → 단일 데이터셋이라 검정력 낮음.
정석은 **데이터셋을 블록**으로 두는 것이므로, CIFAR/network/tabular 결과가 모두 모이면
그때 도메인 통합 스크립트로 재실행할 것.

---

## 실험 결과 (v1, 5 seeds) — ⚠️ gradient 기록 없는 구버전 (8종)

데이터: **IR=328.8:1** (min=39 `grief`, max=12823 `neutral`, 총 36,308 / val 4,548 / test 4,590, vocab 12,226).
Optuna best: `pwce α=0.55`, `plwce α=3.97`, `focal γ=2.05`.

| Loss | Top1 | Balanced | **F1-Macro** | Many | Medium | Few |
|------|------|----------|----------|------|--------|-----|
| ce | 0.5423 | 0.3946 | 0.4084 | 0.5312 | 0.3459 | 0.3154 |
| wce | 0.4042 | 0.4487 | 0.3780 | 0.4989 | 0.3828 | 0.4628 |
| pwce | 0.5312 | 0.4527 | 0.4294 | 0.5434 | 0.4095 | 0.4100 |
| sqce | 0.5232 | 0.4546 | 0.4274 | 0.5369 | 0.4087 | 0.4219 |
| **lwce** | 0.5377 | 0.4209 | 0.4212 | 0.5324 | 0.3705 | 0.3660 |
| **plwce** | 0.5417 | 0.4641 | **0.4380 (1위)** | 0.5372 | 0.3979 | 0.4579 |
| cb | 0.4436 | 0.4939 | 0.4054 | 0.5040 | 0.4167 | **0.5544** |
| focal | 0.5452 | 0.4013 | 0.4149 | 0.5249 | 0.3568 | 0.3301 |

### 주요 소견

- **plwce가 F1-Macro 단독 1위(0.4380)**. vs ce: **Few +0.143 (0.3154→0.4579), Top1 −0.0006(무손실)**.
- **cb/wce의 높은 Few는 head 붕괴의 대가**: cb Few 0.5544지만 Top1 −0.099, wce Few 0.4628지만 Top1 **−0.138**·F1 꼴찌.
  **plwce Few(0.4579) ≈ wce Few(0.4628)인데 plwce만 Top1을 유지** → Proposition 3·4(로그 압축 = 붕괴 없는 소수 클래스 이득)의 직접 실증.
- **plwce(튜닝 α) 강세 / lwce(parameter-free) 보수적** — CIFAR-100(다클래스)와 동일 패턴이 **텍스트+트랜스포머에서 재현** → 아키텍처 독립성 주장 뒷받침.
- ⚠️ 논문 서술 시 cb의 Few 1위를 숨기지 말고 **"Top1 10점 희생의 대가"로 정직하게 프레이밍**할 것.

---

## 알려진 주의사항 (image_classification 상속)

| 항목 | 규칙 |
|------|------|
| Optuna trial 필터 | `if t.value is not None:` |
| GridSampler + MedianPruner | 병행 금지 |
| focal γ 하한 | `1.0` (γ<1 NaN 붕괴) |
| NUM_WORKERS | 0 고정 (Colab) |
| multi-label 처리 | single-label 필터만 (BCE 전환 금지 — CE 기반 LWCE와 불일치) |

---

## 참고 링크

- image_classification/CLAUDE.md — CNN(ResNet-32) 대응 실험, 손실 8종 상세
- paper/main.tex §Experiments — 이 실험이 들어갈 자리(아키텍처 독립성 검증)
