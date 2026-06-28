# CLAUDE.md — network_data

네트워크 침입 탐지(Network Intrusion Detection) 도메인에서 LWCE 계열 손실 함수를 비교하는 실험.
PyTorch MLP 기반.

> **두 가지 실험 트랙이 공존한다:**
> 1. **통합 노트북 (ASC 논문용, 권장)** — `Network_MLP.ipynb` + `scr/data_handler.py`.
>    tabular `Tabular_MLP.ipynb`와 동일한 구조: **8 losses**(ce/wce/pwce/sqce/lwce/plwce/cb/focal, **WCE 포함**),
>    **통일된 Optuna 범위**(PWCE 0.3–5.0, PLWCE 0.5–6.0, **Focal 1.0–5.0**; wce·sqce는 파라미터 없음, sqce=√-CE α=0.5 고정), 5 seeds.
>    **논문 역할**: `lwce`/`plwce` = **proposed**, `sqce` = reported baseline, `pwce` = **분석/이론 섹션용**(α-sweep foil, main 비교 아님), `ce`/`wce`/`cb`/`focal` = baseline.
>    **⚠️ wce 추가 (2026-06, 전 도메인)**: weight-explosion 동기가 정조준하는 표준 IDS 베이스라인이라 비교군에 포함. `wce = pwce(α=1.0)` 특수해(`total/n_i`). **전 도메인 완료 (network·tabular·CIFAR-10·100)**: network·tabular는 wce가 F1 최악(6.12/8·7.18/8, 분리 쉬운 도메인). image LT는 wce가 **IR 의존 붕괴** — CIFAR-10은 IR=10 **rank 1/8(최우수)→IR=100 8/8(꼴찌)** 완벽 단조, CIFAR-100은 5/8→7/8. 모두 weight explosion 실증.
>    한 노트북에서 8개 데이터셋을 `load → Optuna → 학습 → 저장 → 메모리 해제` 루프로 처리.
>    결과: `results/mlp/`. tabular/CIFAR와 cross-domain 비교표를 위해 손실/범위를 통일함.
> 2. **레거시 8개 개별 노트북 (KICS 학술대회용)** — `{Dataset}.ipynb`.
>    7 losses(WCE 포함), 데이터셋별 옛 Optuna 범위. 아래 "실험 결과 요약"은 이 트랙 기준.

## scr/data_handler.py (통합 로더)

- `load_network_dataset(name)` → `(X_tr, X_te, y_tr, y_te, class_counts, num_classes, class_names)`
- tabular `data_handler.load_dataset`와 동일한 train/test 인터페이스 (70/30, NSL-KDD·UNSW는 공식 split)
- generic 6종은 공유 헬퍼(`_detect_files`/`_stratified_subset`/`_finalize`)로, NSL-KDD·UNSW-NB15는
  공식 train/test + 범주형 one-hot 전용 함수로 처리. 8개 노트북의 검증된 Cell 1 전처리를 그대로 포팅.
- Optuna proxy는 노트북에서 `train_test_split(stratify=y_tr)` 계층 추출(ValueError 시 랜덤 fallback).

---

## 모델 및 공통 설정

- **모델**: MLP (256→128→64, BatchNorm, Dropout=0.3)
- **손실 함수**: `ce`, `wce`, `pwce`, `lwce`, `plwce`, `cb`, `focal`
- **평가 지표**: Balanced_Accuracy, F1_Macro, Head_F1, Mid_F1, Tail_F1, F1_Weighted, Accuracy
- **Head/Mid/Tail 경계**: `class_counts`의 33/66 백분위수 기준
- `BATCH_SIZE=512`, `FINAL_EPOCHS=50`, `num_workers=0`

## Optuna 설정

- GridSampler, **20 trials**, **F1-Macro 기준** (`direction='maximize'`)
- proxy: `PROXY_SUBSET_RATIO=0.20`, `PROXY_EPOCHS=10`
- (레거시 8노트북 트랙) PWCE/LWCE/PLWCE alpha 범위: `[1.0, 10.0]` (PWCE는 `[0.5, 5.0]`)
- (통합 노트북 트랙) PWCE `[0.3, 5.0]`, PLWCE `[0.5, 6.0]`, **FOCAL gamma `[1.0, 5.0]`**
- `os.environ['TQDM_DISABLE'] = '1'` 필수

> **⚠️ Focal γ<1 NaN 붕괴 버그 (2026-06 확인, 통합 노트북)**: multiclass softmax focal의
> modulating factor `(1−p_t)^γ` 는 γ<1일 때 gradient `−γ(1−p_t)^(γ−1)`가 `p_t→1`에서 발산.
> 다수 클래스가 confident해지면 gradient 폭발 → NaN → 모든 예측이 class 0으로 collapse
> (`argmax(NaN)=0`). 증상: focal F1≈`F1_class0/K`, **std=0.0000**, Tail_F1=0.0.
> Optuna proxy(10 epoch)는 짧아 낮은 γ를 "최적"으로 잘못 선택 → 본 학습(50 epoch)에서 폭발.
> **해결**: focal gamma grid 하한을 `1.0`으로 제한(γ=0은 CE special-case라 안전, 0<γ<1만 위험).
> 재실행 시 `results/mlp/_best_params.json`과 영향받은 `{Dataset}_results.json` 삭제 필요.

## 데이터셋

| 데이터셋 | Kaggle ID | 클래스 수 | 불균형 | 비고 |
|---------|-----------|---------|------|------|
| NSL-KDD | `hassan06/nslkdd` | 5 | ~1301:1 | KDDTrain+.txt / KDDTest+.txt, 범주형 3개 one-hot |
| CICIDS2017 | `chethuhn/network-intrusion-dataset` | 15 | ~8750:1 | 요일별 CSV 8개, `TARGET_TOTAL=300K`, `BENIGN_MAX=100K` |
| UNSW-NB15 | `mrwellsdavid/unsw-nb15` | 10 | 극심 | training/testing-set CSV, `attack_cat` 기반, `id`·`label` 컬럼 제거 |
| CIC-DDoS2019 | `dhoogla/cicddos2019` | ~18 | ~1957:1 | Parquet 다중 파일, `TARGET_TOTAL=300K`, `BENIGN_MAX=100K` |
| Bot-IoT | `vigneshvenkateswaran/bot-iot` | 4 | ~1366:1 | `category` 컬럼(DDoS/DoS/Normal/Reconnaissance), 75 CSV 대용량 → 서브샘플링. Normal 54 최소 |
| TON_IoT | `arnobbhowmik/ton-iot-network-dataset` | 10 | ~48:1 | `train_test_network.csv`, `type` 컬럼(10-class). normal 35K가 최다, mitm 730이 최소 |
| CICIDS2018 | `solarmainframe/ids-intrusion-csv` | 15 | ~17,500:1 | 요일별 CSV, `Label` 컬럼 strip() 필수. Benign 70K vs SQL Injection 4 |
| RT-IoT2022 | `supplejade/rt-iot2022real-time-internet-of-things` | 12 | ~3,313:1 | `RT_IOT2022.csv`, `Attack_type` 컬럼, ~123K rows 전체 사용. DOS_SYN_Hping 76.9% 지배(IR 높아도 분류 쉬움) |

## Bot-IoT / TON_IoT / CICIDS2018 / RT-IoT2022 공통 주의사항

- Kaggle ID가 불확실한 경우 Cell 1 실행 시 `All files found` 목록이 출력됨 → 실제 파일명 확인 후 Cell 0의 ID 수정
- 레이블 컬럼 자동 감지: `LABEL_PRIORITY` 순서대로 탐색, 없으면 마지막 컬럼 사용
- `_detect_files()` / `_load_files()` / `_stratified_subset()` 헬퍼 함수가 Cell 1 상단에 정의됨

### Bot-IoT 전처리 주의사항
- `category` 컬럼 우선. 전체 라벨은 5종(Normal, DDoS, DoS, Reconnaissance, Theft)이나
  Theft가 극소수라 서브샘플(592K)에선 누락 → **실측 4-class**(Normal, DDoS, DoS, Reconnaissance)
- `attack` 컬럼(이진)이 선택되면 경고 출력 → `category`/`sub_category` 확인 필요
- 파일명에 `feature` 포함된 CSV는 로딩 제외 (피처명 정의 파일)

### TON_IoT 전처리 주의사항
- Network_dataset 파일 우선 (`network` in filename); 없으면 전체 파일 사용
- `type` 컬럼 소문자 정규화 적용 (`.str.lower()`)

### CICIDS2018 전처리 주의사항
- CICIDS2017과 동일 패턴: `Label` 컬럼 strip() 후 LabelEncoder

### RT-IoT2022 전처리 주의사항
- 데이터셋 크기가 TARGET_TOTAL(300K) 이하이면 서브샘플링 생략
- `Attack_type` 컬럼 우선

## UNSW-NB15 전처리 주의사항

- `attack_cat`에 선행/후행 공백 존재 → `.str.strip()` 필수
- 정상 트래픽은 `attack_cat = ''` (빈 문자열) → `'Normal'`로 치환
- 범주형 피처: `proto`, `service`, `state` → one-hot (train+test 합쳐서 fit)
- 제거 컬럼: `id` (식별자), `label` (이진 레이블, `attack_cat`으로 대체)
- pre-defined train/test split 사용 — `TARGET_TOTAL` 서브샘플링 불필요

## CIC-DDoS2019 전처리 주의사항

- 레이블 컬럼: 공백 포함 가능 → `.str.strip()` 후 LabelEncoder
- BENIGN 클래스 상한 `BENIGN_MAX=100K` 적용 (CICIDS2017 동일 패턴)
- Kaggle ID가 불확실하면 노트북 Cell 0 주석 참조

## 노트북 셀 구성 (표준)

| 셀 번호 | 내용 |
|--------|------|
| Cell 0 | 환경설정, 상수 (`DATASET_NAME`, `NUM_CLASSES`, `RESULTS_DIR`) |
| Cell 1 | 데이터 로드 + `make_loaders()` |
| Cell 2 | 클래스 분포 시각화 |
| Cell 3 | MLP 모델 + `compute_metrics()` |
| Cell 4 | `train_model()` 정의 (호출 금지) |
| Cell 5 | Optuna 탐색 (PWCE / PLWCE / FOCAL) |
| Cell 6 | 전체 실험 실행 |
| Cell 7 | 시각화 + JSON/Excel 저장 |

## 결과 저장 경로

```
/content/gdrive/MyDrive/imbalanced-data-LWCE/network_data/results/{Dataset}/
```

## (레거시·아카이브) 실험 결과 — KICS 8노트북 트랙 (wce 포함, 2026-05-06, N=5)

> 아래는 레거시 8개 개별 노트북(wce 포함, 옛 전처리) 결과. **ASC 논문은 맨 아래 "통합 노트북 트랙"을 사용**하며,
> 두 트랙은 전처리·클래스 수·불균형비가 달라 수치가 직접 비교되지 않는다(예: TON_IoT legacy F1 ~0.69 vs 통합 ~0.39).
>
> **⚠️ 레거시 focal UNSW-NB15 Tail_F1=0.000 = γ<1 NaN 붕괴 아티팩트** (원인·해결은 위 L42–47 참조). 레거시는
> focal γ 탐색 하한이 1.0 미만이라 이 버그가 그대로 박혀 있음. 통합 트랙에서 γ≥1.0으로 정정 → UNSW focal Tail_F1≈0.153.
> **KICS 프로시딩(`KICS 하계학술대회/2026 KICS_하계학술대회_proceeding_고승호.pdf`)은 이미 0.000 수치로 제출 완료** —
> 따라서 **포스터(`KICS2026_LWCE_poster.tex`)는 레거시 트랙·손실 세트(WCE 포함)를 그대로 유지**하되,
> 방어 불가능한 "0.000" 문구만 제거하고 *"Focal은 γ-민감, tail에서 최저"* 로 정성 서술함(2026-06 결정, Option B).
> 별개 run인 통합 트랙의 0.153을 레거시 표에 끼워넣지 않음(손실 세트 WCE↔sqce 상이로 혼입 시 부정직).

### 전체 요약 (F1-Macro 기준)

| 데이터셋 | 클래스 수 | 불균형비 | 🥇 최우수 (F1-Macro) | CE 대비 | 비고 |
|---------|---------|--------|-------------------|--------|------|
| NSL-KDD | 5 | ~1301:1 | `plwce` (0.6376) | +0.039 | lwce도 근접 (0.6349) |
| CICIDS2017 | 15 | ~8750:1 | `ce` (0.8082) | — | plwce가 Bal_Acc 최우수 (0.8510) |
| UNSW-NB15 | 10 | ~850:1 | `wce` (0.4579) | +0.046 | focal Tail_F1=0.000 붕괴 |
| CIC-DDoS2019 | 18 | ~1957:1 | `cb` (0.5026) | +0.051 | wce Bal_Acc 최우수 (0.6012) |
| Bot-IoT | 4 | ~1366:1 | `wce` (0.9151) | +0.090 | wce Tail_F1도 최우수 (0.6635) |
| TON_IoT | 10 | ~48:1 | `pwce` (0.7302) | +0.036 | lwce Tail_F1 최우수 (0.3516) |
| CICIDS2018 | 15 | ~17,500:1 | `lwce` (0.7846) | +0.063 | lwce Tail_F1도 최우수 (0.6148) |
| RT-IoT2022 | 12 | ~3,313:1 | `lwce` (0.9632) | +0.006 | IR 높아도 다수 클래스 76.9% 지배 → 분리 쉬워 전반 고성능 |

### 상세 결과 (mean ± std)

#### NSL-KDD (~1301:1)
| Loss | Bal_Acc | F1_Macro | Head_F1 | Tail_F1 |
|------|---------|----------|---------|---------|
| ce    | 0.5688±0.0169 | 0.5987±0.0204 | 0.8489±0.0043 | 0.2693±0.0538 |
| wce   | 0.6205±0.0083 | 0.5840±0.0201 | 0.8552±0.0021 | 0.2201±0.0463 |
| pwce  | 0.6152±0.0140 | 0.5891±0.0142 | 0.8540±0.0079 | 0.2409±0.0309 |
| lwce  | 0.6008±0.0037 | 0.6349±0.0028 | 0.8505±0.0021 | 0.3513±0.0080 |
| **plwce** | **0.6081±0.0090** | **0.6376±0.0098** | 0.8522±0.0039 | **0.3649±0.0098** |
| cb    | 0.6201±0.0074 | 0.5953±0.0141 | 0.8523±0.0024 | 0.2506±0.0366 |
| focal | 0.5733±0.0072 | 0.6073±0.0082 | 0.8459±0.0016 | 0.2908±0.0191 |

#### CICIDS2017 (~8750:1)
| Loss | Bal_Acc | F1_Macro | Head_F1 | Tail_F1 |
|------|---------|----------|---------|---------|
| **ce**    | 0.8191±0.0006 | **0.8082±0.0029** | **0.9900±0.0010** | **0.4965±0.0085** |
| wce   | 0.8377±0.0018 | 0.7412±0.0062 | 0.9757±0.0024 | 0.4098±0.0192 |
| pwce  | 0.8408±0.0006 | 0.7741±0.0049 | 0.9851±0.0010 | 0.4511±0.0081 |
| lwce  | 0.8177±0.0095 | 0.7976±0.0070 | 0.9896±0.0013 | 0.4596±0.0244 |
| plwce | **0.8510±0.0075** | 0.7818±0.0098 | 0.9816±0.0019 | 0.4909±0.0324 |
| cb    | 0.8390±0.0144 | 0.7711±0.0144 | 0.9842±0.0009 | 0.4475±0.0444 |
| focal | 0.8174±0.0017 | 0.8008±0.0036 | 0.9864±0.0019 | 0.4830±0.0116 |

#### UNSW-NB15 (~850:1)
| Loss | Bal_Acc | F1_Macro | Head_F1 | Tail_F1 |
|------|---------|----------|---------|---------|
| ce    | 0.4116±0.0039 | 0.4120±0.0042 | 0.6542±0.0049 | 0.1214±0.0098 |
| **wce**   | **0.5598±0.0038** | **0.4579±0.0021** | **0.6900±0.0057** | 0.1931±0.0026 |
| pwce  | 0.4938±0.0249 | 0.4574±0.0068 | 0.6617±0.0032 | **0.2364±0.0161** |
| lwce  | 0.4351±0.0057 | 0.4338±0.0069 | 0.6599±0.0027 | 0.1725±0.0236 |
| plwce | 0.4890±0.0280 | 0.4483±0.0072 | 0.6498±0.0060 | 0.2285±0.0112 |
| cb    | 0.5372±0.0084 | 0.4493±0.0041 | 0.6534±0.0046 | 0.2127±0.0086 |
| focal | 0.3577±0.0108 | 0.3494±0.0122 | 0.6648±0.0100 | 0.0000±0.0000 |

#### CIC-DDoS2019 (~1957:1)
| Loss | Bal_Acc | F1_Macro | Head_F1 | Tail_F1 |
|------|---------|----------|---------|---------|
| ce    | 0.4984±0.0036 | 0.4515±0.0051 | 0.7848±0.0004 | 0.1170±0.0161 |
| wce   | **0.6012±0.0088** | 0.4938±0.0019 | **0.8064±0.0074** | 0.2026±0.0084 |
| pwce  | 0.5686±0.0059 | 0.5015±0.0030 | 0.7851±0.0006 | 0.2106±0.0302 |
| lwce  | 0.5266±0.0032 | 0.4841±0.0075 | 0.7852±0.0003 | 0.1941±0.0164 |
| plwce | 0.5641±0.0091 | 0.4922±0.0047 | 0.7843±0.0004 | 0.2102±0.0221 |
| **cb**    | 0.5760±0.0017 | **0.5026±0.0040** | 0.7870±0.0022 | **0.2539±0.0105** |
| focal | 0.5057±0.0035 | 0.4568±0.0045 | 0.7851±0.0002 | 0.1267±0.0125 |

#### Bot-IoT (~1366:1)
| Loss | Bal_Acc | F1_Macro | Head_F1 | Tail_F1 |
|------|---------|----------|---------|---------|
| ce    | 0.7954±0.0001 | 0.8250±0.0023 | 0.9999±0.0000 | 0.3033±0.0088 |
| **wce**   | **0.8772±0.0182** | **0.9151±0.0167** | 1.0000±0.0000 | **0.6635±0.0668** |
| pwce  | 0.8317±0.0181 | 0.8682±0.0190 | 1.0000±0.0000 | 0.4763±0.0761 |
| lwce  | 0.8499±0.0369 | 0.8844±0.0394 | 1.0000±0.0000 | 0.5410±0.1577 |
| plwce | 0.8454±0.0333 | 0.8821±0.0317 | 1.0000±0.0000 | 0.5317±0.1269 |
| cb    | 0.8587±0.0338 | 0.8930±0.0365 | 0.9995±0.0010 | 0.5764±0.1443 |
| focal | 0.8454±0.0265 | 0.8836±0.0277 | 1.0000±0.0000 | 0.5376±0.1106 |

#### TON_IoT (극심)
| Loss | Bal_Acc | F1_Macro | Head_F1 | Tail_F1 |
|------|---------|----------|---------|---------|
| ce    | 0.6902±0.0505 | 0.6940±0.0522 | 0.7543±0.0554 | 0.1511±0.0571 |
| wce   | 0.7447±0.0450 | 0.6720±0.0469 | 0.7305±0.0480 | 0.1461±0.0453 |
| **pwce**  | **0.8002±0.0399** | **0.7302±0.0403** | **0.7908±0.0405** | 0.1850±0.0536 |
| lwce  | 0.7178±0.0649 | 0.7082±0.0562 | 0.7478±0.0591 | **0.3516±0.0525** |
| plwce | 0.7708±0.0388 | 0.7093±0.0359 | 0.7590±0.0386 | 0.2617±0.0638 |
| cb    | 0.7846±0.0579 | 0.7216±0.0633 | 0.7764±0.0651 | 0.2277±0.0551 |
| focal | 0.6853±0.0530 | 0.6813±0.0601 | 0.7354±0.0601 | 0.1947±0.1052 |

#### CICIDS2018 (극심)
| Loss | Bal_Acc | F1_Macro | Head_F1 | Tail_F1 |
|------|---------|----------|---------|---------|
| ce    | 0.7173±0.0103 | 0.7220±0.0138 | 0.8810±0.0005 | 0.4289±0.0475 |
| wce   | 0.7683±0.0006 | 0.7601±0.0047 | 0.8726±0.0130 | 0.5585±0.0078 |
| pwce  | 0.7654±0.0006 | 0.7716±0.0035 | 0.8817±0.0010 | 0.5898±0.0133 |
| **lwce**  | **0.7641±0.0003** | **0.7846±0.0032** | 0.8813±0.0003 | **0.6148±0.0089** |
| plwce | 0.7641±0.0004 | 0.7754±0.0024 | 0.8809±0.0006 | 0.5900±0.0103 |
| cb    | 0.7634±0.0007 | 0.7708±0.0037 | 0.8784±0.0017 | 0.5849±0.0137 |
| focal | 0.7570±0.0135 | 0.7755±0.0193 | 0.8804±0.0003 | 0.5943±0.0539 |

#### RT-IoT2022 (중간)
| Loss | Bal_Acc | F1_Macro | Head_F1 | Tail_F1 |
|------|---------|----------|---------|---------|
| ce    | 0.9491±0.0020 | 0.9574±0.0010 | 0.9871±0.0028 | 0.8957±0.0029 |
| wce   | 0.9653±0.0016 | 0.9288±0.0032 | 0.9832±0.0024 | 0.8158±0.0070 |
| pwce  | **0.9675±0.0016** | 0.9565±0.0057 | 0.9867±0.0024 | 0.8955±0.0178 |
| **lwce**  | 0.9560±0.0029 | **0.9632±0.0050** | **0.9895±0.0010** | **0.9105±0.0141** |
| plwce | 0.9588±0.0045 | 0.9606±0.0068 | 0.9890±0.0010 | 0.9037±0.0197 |
| cb    | 0.9618±0.0028 | 0.9350±0.0125 | 0.9788±0.0051 | 0.8417±0.0340 |
| focal | 0.9554±0.0044 | 0.9612±0.0051 | 0.9876±0.0009 | 0.9068±0.0143 |

### 전체 소견

- **lwce/plwce 우세**: 5/8 데이터셋에서 상위권 (NSL-KDD, CICIDS2018, RT-IoT2022, TON_IoT Tail, CIC-DDoS2019 근접)
- **CICIDS2017 예외**: 극심한 불균형(8750:1)이지만 CE가 F1-Macro 최우수 — 가중치 손실이 소수 클래스를 과하게 부스팅하면서 다수 클래스 F1이 떨어지는 트레이드오프
- **focal 불안정**: UNSW-NB15에서 Tail_F1=0.000 완전 붕괴, CIC-DDoS2019·NSL-KDD에서도 중위권
- **wce 강세**: Bot-IoT(4-class 단순 구조)에서 wce가 압도적 우위 (+0.090)
- **불균형 ≤ 중간(RT-IoT2022)**: 모든 손실 함수가 0.93+ F1-Macro로 차이 미미

## 실험 결과 요약 — 통합 노트북 트랙 (ASC 논문용, 2026-06, N=5)

수치 출처: `_checkpoint_network.json` (**8 losses, wce 포함**, sqce 포함, 2026-06 재실행). **lwce/plwce가 proposed.**
F1-Macro 기준 8종 중 순위(`n/8`)로 표기. 통합 IR: NSL-KDD 1295:1, UNSW 841:1, CICIDS2017 8750:1, CICIDS2018 17,500:1, CIC-DDoS2019 1957:1, Bot-IoT 1366:1, TON_IoT 48:1, RT-IoT2022 3313:1.

**🔑 wce(역빈도) = 8종 중 최악** — 평균순위 F1 **6.12/8**, Tail **7.00/8(사실상 꼴찌)**. 극심 불균형일수록 붕괴: CICIDS2018(17,500:1)·CICIDS2017(8750:1)·RT-IoT2022(3313:1)·Bot-IoT(1366:1)에서 **F1·Tail 모두 단독 8/8**. Tail-F1이 NSL 0.1445(vs lwce 0.5256)·RT22 0.6415(vs lwce 0.8742)로 붕괴. **weight explosion을 실증** — bounded reweighting(lwce/plwce/sqce/pwce)이 unbounded wce를 일관 상회. 유일 예외 CIC-DDoS2019(wce F1 1위 0.4753, but Tail 5/8). 8종 평균순위: pwce 3.00 / plwce 3.25 / lwce 3.75 / sqce 4.25 / focal 4.75 / ce 5.00 / cb 5.75 / **wce 6.12**.

| 데이터셋 | 🥇 최우수 (F1) | **lwce** | **plwce** | wce | 소수클래스(Tail) |
|---------|---------------|----------|-----------|-----|------------------|
| NSL-KDD | lwce 0.6349 | **0.6349 (1/8) 🥇** | 0.6343 (2/8) | 0.5778 (6/8) | **plwce Tail 0.5457 최우수** |
| UNSW-NB15 | pwce 0.4624 | 0.4439 (4/8) | 0.4594 (2/8) | 0.4399 (5/8) | pwce Tail 0.2498 최상 |
| CICIDS2017 | plwce 0.7866 | 0.7829 (2/8) | **0.7866 (1/8) 🥇** | 0.6482 (8/8) | pwce Tail 0.4888 최상 |
| CICIDS2018 | pwce 0.8046 | 0.8042 (2/8) | 0.8031 (3/8) | 0.6741 (8/8) | **lwce·plwce Tail 0.6760 공동 최상** |
| CIC-DDoS2019 | **wce 0.4753** | 0.4471 (7/8) | 0.4626 (5/8) | **0.4753 (1/8)** | **plwce Tail 0.1869 최우수** |
| Bot-IoT | pwce/sqce 0.9182 | 0.9147 (5/8) | 0.9106 (6/8) | 0.9070 (8/8) | 전부 노이즈 내(±0.01) |
| TON_IoT | ce 0.3933 | 0.3590 (6/8) | 0.3887 (2/8) | 0.3593 (5/8) | plwce Tail 0.5508 최상(고분산) |
| RT-IoT2022 | ce 0.9544 | 0.9512 (3/8) | 0.9292 (5/8) | 0.8696 (8/8) | ce Tail 0.8838 최상 |

**제안 손실 소견 (lwce/plwce)**
- **상위권 다수**: 단독 1위 2/8(NSL-KDD lwce, CICIDS2017 plwce), **top-2 진입 4/8**(+UNSW-NB15 plwce, CICIDS2018 lwce). CICIDS2018은 pwce/lwce/plwce가 0.803~0.805로 사실상 동률.
- **Tail(소수 클래스)이 핵심 강점**: plwce가 NSL-KDD(0.5457)·CIC-DDoS2019(0.1869) Tail 최우수, lwce·plwce가 CICIDS2018 Tail 공동 최상(0.6760). F1-Macro에서 안 보여도 소수 클래스 회복은 제안 손실이 자주 1위.
- **lwce vs plwce 상보적**: plwce는 Tail·중간 불균형(UNSW/TON_IoT 2위)에서, lwce는 다클래스 균형 F1(NSL-KDD 1위)에서 우위.
- **약세 케이스**: 극단 few-class 불균형(CIC-DDoS2019는 wce, Bot-IoT는 pwce/sqce)에 밀림 / ce가 이미 강한 TON_IoT·RT-IoT2022는 ce 우위.

> **세 도메인 통합 narrative**: 제안 손실(특히 plwce)의 강점은 **소수 클래스 Tail 회복**과 **다클래스·중간 불균형**.
> 파라미터 없는 √-CE(sqce)는 **few-class 극단 불균형**(CIC-DDoS2019, CIFAR-10 고IR)에서 빛남 → 역할 구분.
> CB는 다수 도메인에서 변동성·최하위 빈번. cf. `image_classification/CLAUDE.md`, `tabular_data/CLAUDE.md` 동일 결론.
