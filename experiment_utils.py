"""
3차 피드백(2026-07-12) 대응 공용 계측 유틸 — image / tabular / network / text 4개 도메인 공유.

custom_losses.py와 함께 **리포 루트**에 둔다 (Colab: /content/drive/MyDrive/imbalanced-data-LWCE).
각 노트북 Cell 0은 `sys.path.insert(0, REPO)` 를 마지막에 호출해 루트를 최우선 경로로 만든다.

제공 기능:
    group_masks()      — Many/Few tertile 마스크 (전 도메인 동일 정의; 이진 클래스 대응)
    total_grad_norm()  — 클리핑 없이 파라미터 gradient의 total L2 norm
    GradLogger         — epoch별 grad_norm / grad_many / grad_few / grad_ratio 집계
    extended_metrics() — G-Mean, Worst-class Acc, Per-class Recall (§4.3 지표)
    OPTUNA_GRIDS       — 전 도메인 통일 탐색 그리드

왜 공용 모듈인가: 같은 계측 코드를 4개 노트북에 복제하면 도메인 간 미묘한 정의 차이가 생겨
gradient 수치를 서로 비교할 수 없게 된다. 논문이 도메인 간 비교를 하므로 정의는 하나여야 한다.
"""

import numpy as np
import torch


# ==================================================================
# 그룹 정의 (Many / Few) — 전 도메인 동일
# ==================================================================
def group_masks(class_counts, num_classes=None):
    """
    train count 기준 tertile split. CIFAR/GoEmotions 기존 관례와 동일:
        many = 상위 n//3,  few = 하위 (n - 2*(n//3))

    Returns: (is_many, is_few) — 각각 (num_classes,) bool 텐서

    주의: 이진 분류(n=2)에서는 n//3 == 0 이라 위 공식이 빈 마스크를 만든다.
          tabular에 이진 데이터셋이 여럿 있으므로 반드시 특수 처리.
    """
    counts = np.asarray(class_counts)
    n = len(counts)
    if num_classes is None:
        num_classes = n
    # [::-1]은 음수 stride 뷰라 torch가 거부한다 → 여기서 한 번 copy
    order = np.argsort(counts)[::-1].copy()   # many -> few

    if n < 3:
        many_idx, few_idx = order[:1], order[1:]
    else:
        many_idx, few_idx = order[:n // 3], order[2 * (n // 3):]

    is_many = torch.zeros(num_classes, dtype=torch.bool)
    is_few = torch.zeros(num_classes, dtype=torch.bool)
    is_many[torch.as_tensor(many_idx, dtype=torch.long)] = True
    is_few[torch.as_tensor(few_idx, dtype=torch.long)] = True
    return is_many, is_few


# ==================================================================
# Gradient 계측 (3차 피드백 §4.7)
# ==================================================================
def total_grad_norm(model):
    """
    파라미터 gradient의 total L2 norm. clip_grad_norm_과 달리 grad를 수정하지 않으므로
    클리핑을 쓰지 않는 노트북(CIFAR SGD, network/tabular Adam)에서도 안전하게 쓸 수 있다.
    """
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().norm().item()) ** 2
    return sq ** 0.5


class GradLogger:
    """
    epoch별 gradient 통계 집계기.

    사용법 (어떤 학습 루프에도 동일):
        glog = GradLogger(class_counts, num_classes, device)
        for epoch in ...:
            glog.reset()
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                logits.retain_grad()              # ← 필수
                loss = criterion(logits, yb)
                loss.backward()
                glog.update(logits, yb, model)    # optimizer.step() 전에 호출
                optimizer.step()
            for k, v in glog.epoch_end().items():
                history[k].append(v)

    측정값:
      grad_norm  — 파라미터 gradient total norm (Optimization Stability)
      grad_many / grad_few — 샘플별 '로짓' gradient norm의 그룹 평균.
                    로짓 기울기는 w_{y_i}(p_i - e_{y_i}) 이므로 Proposition 4와 직접 대응한다.
      grad_ratio — grad_few / grad_many. 배치 정규화 상수(예: nn.CrossEntropyLoss의
                   weighted-mean 정규화 1/Σw)는 분자·분모에서 상쇄되므로 scale-invariant.
    """

    def __init__(self, class_counts, num_classes, device):
        is_many, is_few = group_masks(class_counts, num_classes)
        self.is_many = is_many.to(device)
        self.is_few = is_few.to(device)
        self.reset()

    def reset(self):
        self._gn_sum = 0.0
        self._n_batch = 0
        self._many_sum = 0.0
        self._many_n = 0
        self._few_sum = 0.0
        self._few_n = 0

    def update(self, logits, yb, model=None, grad_norm=None):
        """
        loss.backward() 직후, optimizer.step() 전에 호출.
        grad_norm을 직접 주면(예: clip_grad_norm_의 리턴값) 재계산하지 않는다.
        """
        if grad_norm is None:
            if model is None:
                raise ValueError('model 또는 grad_norm 중 하나는 필요합니다.')
            grad_norm = total_grad_norm(model)
        self._gn_sum += float(grad_norm)
        self._n_batch += 1

        if logits.grad is None:      # retain_grad() 누락 시 조용히 틀리는 것보다 즉시 실패
            raise RuntimeError('logits.grad가 None입니다. loss.backward() 전에 '
                               'logits.retain_grad()를 호출했는지 확인하세요.')
        with torch.no_grad():
            gs = logits.grad.norm(dim=1)
            m_mask, f_mask = self.is_many[yb], self.is_few[yb]
            if m_mask.any():
                self._many_sum += float(gs[m_mask].sum())
                self._many_n += int(m_mask.sum())
            if f_mask.any():
                self._few_sum += float(gs[f_mask].sum())
                self._few_n += int(f_mask.sum())

    def epoch_end(self):
        many = self._many_sum / max(1, self._many_n)
        few = self._few_sum / max(1, self._few_n)
        return {
            'grad_norm': self._gn_sum / max(1, self._n_batch),
            'grad_many': many,
            'grad_few': few,
            'grad_ratio': (few / many) if many > 0 else float('nan'),
        }


# ==================================================================
# 평가 지표 (3차 피드백 §4.3)
# ==================================================================
def extended_metrics(y_true, y_pred, num_classes):
    """
    Per-class Recall 기반 지표. G-Mean과 Worst-class Acc는 recall(=class-wise accuracy)로
    정의하는 것이 표준이므로, per-class F1이 아니라 recall을 쓴다.

    Returns: dict(Per_Class_Acc, G_Mean, Worst_Acc, Balanced_Acc)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rec = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        m = (y_true == c)
        rec[c] = float((y_pred[m] == c).mean()) if m.sum() > 0 else 0.0
    return {
        'Per_Class_Acc': rec.tolist(),
        # 한 클래스라도 recall=0이면 G-Mean=0 (최악 클래스에 민감한 것이 이 지표의 목적)
        'G_Mean': float(np.exp(np.mean(np.log(rec + 1e-12)))),
        'Worst_Acc': float(rec.min()),
        'Balanced_Acc': float(rec.mean()),
    }


# ==================================================================
# Optuna 그리드 — 전 도메인 통일 (3차 피드백 §3.3 재현성)
# ==================================================================
# n_trials는 반드시 grid 크기와 일치시킬 것 (GridSampler 규칙).
# eps는 효과가 배수적이라 logspace. focal gamma 하한 1.0 (γ<1 NaN 붕괴).
# 1D는 전 도메인 30 trials로 통일. combined만 2D라 2배(60 = 10 x 6).
OPTUNA_GRIDS = {
    'pwce':     {'alpha': np.linspace(0.3, 5.0, 30).tolist()},
    'plwce':    {'alpha': np.linspace(0.5, 6.0, 30).tolist()},
    'focal':    {'gamma': np.linspace(1.0, 5.0, 30).tolist()},
    'eslwce':   {'eps':   np.logspace(-1, 1, 30).tolist()},
    'logitadj': {'tau':   np.linspace(0.25, 2.0, 30).tolist()},
    'combined': {'alpha': np.linspace(0.5, 6.0, 10).tolist(),
                 'eps':   np.logspace(-1, 1, 6).tolist()},      # 2D 10 x 6 = 60
}

# 전 도메인 공통 손실 목록 (8 기존 + eslwce/combined/logitadj)
LOSS_CONFIGS_11 = ['ce', 'wce', 'pwce', 'sqce', 'lwce', 'plwce',
                   'eslwce', 'combined', 'cb', 'focal', 'logitadj']

# 3차 피드백 §4.5 ablation 표: (loss, LogCompression, Power alpha, Smoothing eps)
ABLATION_ROWS = [('wce', 'X', 'X', 'X'), ('lwce', 'O', 'X', 'X'), ('plwce', 'O', 'O', 'X'),
                 ('eslwce', 'O', 'X', 'O'), ('combined', 'O', 'O', 'O')]


def grid_n_trials(grid):
    """GridSampler에 넘길 n_trials = grid 크기 (정확히 일치해야 함)."""
    return int(np.prod([len(v) for v in grid.values()]))
