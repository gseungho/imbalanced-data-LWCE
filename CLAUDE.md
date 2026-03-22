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

### 노트북 구조
- 모든 노트북은 `medical_data/` 하위에 위치
- 셀 구성: Markdown 헤더 → Cell 0(환경설정) → Cell 1(데이터) → Cell 2(모델) → Cell 3(학습함수) → Cell 4(Optuna) → Cell 5(학습실행) → Cell 6(평가/저장)
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
