# CLAUDE.md — network_data

네트워크 침입 탐지(Network Intrusion Detection) 도메인에서 LWCE 계열 손실 함수를 비교하는 실험.
PyTorch MLP 기반, 7가지 손실 함수 × N 데이터셋 비교.

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
- PWCE alpha 범위: `[0.5, 5.0]`
- PWCE/LWCE/PLWCE alpha 범위: `[1.0, 10.0]` (PWCE는 `[0.5, 5.0]`)
- FOCAL gamma 범위: `[0.5, 5.0]`
- `os.environ['TQDM_DISABLE'] = '1'` 필수

## 데이터셋

| 데이터셋 | Kaggle ID | 클래스 수 | 불균형 | 비고 |
|---------|-----------|---------|------|------|
| NSL-KDD | `hassan06/nslkdd` | 5 | ~1301:1 | KDDTrain+.txt / KDDTest+.txt, 범주형 3개 one-hot |
| CICIDS2017 | `chethuhn/network-intrusion-dataset` | 15 | ~8750:1 | 요일별 CSV 8개, `TARGET_TOTAL=300K`, `BENIGN_MAX=100K` |
| UNSW-NB15 | `mrwellsdavid/unsw-nb15` | 10 | 극심 | training/testing-set CSV, `attack_cat` 기반, `id`·`label` 컬럼 제거 |
| CIC-DDoS2019 | `dhoogla/cicddos2019` *(확인 필요)* | ~13 | 극심 | 다중 CSV, `TARGET_TOTAL=300K`, `BENIGN_MAX=100K` |

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

## 실험 결과 요약 (2026-05-03 기준)

| 데이터셋 | 불균형 | 🥇 최우수 | CE 대비 F1-Macro |
|---------|--------|----------|----------------|
| NSL-KDD | ~1301:1 | `plwce` | +0.028 |
| CICIDS2017 | ~8750:1 | `plwce` | +0.049 |
| UNSW-NB15 | — | — | 진행 중 |
| CIC-DDoS2019 | — | — | 진행 중 |
