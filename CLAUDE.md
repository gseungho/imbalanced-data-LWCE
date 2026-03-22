# 프로젝트 컨텍스트

불균형 클래스 환경에서 의료 이미지 세그멘테이션 성능을 검증하는 연구 프로젝트.
**커스텀 손실 함수(LWCE 계열)**를 다양한 의료 영상 도메인에 적용·비교함.

자세한 규칙, 실험 설계, 도메인 정보는 `@medical_data/rules.md` 참조.

---

## 핵심 규칙 요약

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
  - **LBL (Log-Boundary Loss)**: 거리 맵에 `log(1+|d|)` 적용 — LWCE의 log scaling 철학을 픽셀 거리 도메인에 확장, "쉬운 배경 픽셀" 비중 감소 효과

### 노트북 구조
- 도메인 실험: `medical_data/` 하위 위치
- Boundary ablation: `medical_data/boundary_ablation/` 하위 위치 (Optuna 포함, Cell 0~7)
- 셀 구성: Markdown 헤더 → Cell 0(환경설정) → Cell 1(데이터) → Cell 2(클래스비율) → Cell 3(모델) → Cell 4(학습함수) → Cell 5(Optuna) → Cell 6(학습실행) → Cell 7(평가/저장)
- 각 코드 셀 첫 줄: `# === Cell N: 셀 이름 ===`
- 섹션 구분 주석: `# --- 소제목 ---`

### 경로 규칙
- custom_losses import: `sys.path.insert(0, '/root/imbalanced-data-LWCE/medical_data')`
- 오타 주의: `imbalanced` (올바름) vs `inbalanced` (오타, 기존 일부 파일에 존재)
- 결과 저장: `medical_data/results/{도메인}/`
- 임시 파일(체크포인트, 캐시): `/tmp/`

### Optuna
- TQDM_DISABLE 환경변수로 trial 중 tqdm 출력 억제: `os.environ['TQDM_DISABLE'] = '1'`
- `if t.value is not None` — `if t.value` 사용 금지 (Dice=0.0 trial 필터링 버그)
- study_pf 시각화 코드는 반드시 `study_pf = optuna.create_study(...)` 정의 이후에 위치

### 데이터 로딩
- WMH: `kagglehub.dataset_download("farahmo/wmh-dataset")` 자동 다운로드
- Pancreas: Google Drive `synapse/synapse.zip` → `/tmp/` 압축 해제
- 기타: Google Drive `MyDrive/imbalanced-data-LWCE/{domain}/` → `/tmp/` 복사

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
