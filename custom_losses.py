"""
분류용 손실함수 11종 — image / tabular / network / text 4개 도메인 공유.

리포 루트에 둔다 (Colab: /content/drive/MyDrive/imbalanced-data-LWCE).
각 노트북 Cell 0은 `sys.path.insert(0, REPO)` 를 마지막에 호출해 루트를 최우선 경로로 만든다.

⚠️ 이름이 같은 다른 모듈이 두 개 있으니 혼동 금지:
   - tabular_data/src/custom_losses.py — XGBoost용 numpy grad/hess (레거시 KIIS 트랙)
   - medical_data/custom_losses.py     — 세그멘테이션용 (get_loss_function API)
   이 파일은 backbone-agnostic 분류용이며 팩토리는 get_clf_loss().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# ==================================================================
# 가중치 계산 함수
# ==================================================================
def calculate_weights(class_counts, mode='ce', alpha=1.0, beta=0.9999, eps=0.1):
    """
    클래스 불균형을 보정하기 위한 가중치 계산.

    Args:
        class_counts: 각 클래스의 샘플 수 리스트
        mode: 'ce' (균일), 'wce' (역빈도), 'sqce' (역제곱근 빈도),
              'lwce' (로그 가중치), 'plwce' (파워 로그 가중치),
              'eslwce' (eps 스무딩 로그 가중치), 'combined' (plwce+eslwce), 'cb' (클래스 균형)
        alpha: plwce / combined 지수
        beta: cb 베타 파라미터
        eps: eslwce / combined 스무딩 상수 (가중치를 1/eps로 상한)

    Returns:
        torch.Tensor: 정규화된 가중치
    """
    counts = np.array(class_counts, dtype=np.float32)
    total = np.sum(counts)
    safe_counts = counts + 1e-6

    if mode == 'ce':
        weights = np.ones_like(counts)
    elif mode == 'wce':
        weights = total / safe_counts
    elif mode == 'pwce':
        weights = np.power(total / safe_counts, alpha)
    elif mode == 'sqce':
        weights = np.sqrt(total / safe_counts)
    elif mode == 'lwce':
        weights = 1.0 / np.log1p(safe_counts)
    elif mode == 'plwce':
        weights = 1.0 / np.power(np.log1p(safe_counts), alpha)
    elif mode == 'eslwce':
        # n_c=0 에서도 유한 (분모 >= eps) → 가중치 상한 1/eps
        weights = 1.0 / (np.log1p(safe_counts) + eps)
    elif mode == 'combined':
        # PLWCE(alpha) ∘ ES-LWCE(eps) — orthogonality ablation용
        # alpha=1 → eslwce, eps=0 → plwce 로 환원됨
        weights = 1.0 / np.power(np.log1p(safe_counts) + eps, alpha)
    elif mode == 'cb':
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts) + 1e-6)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


# ==================================================================
# Focal Loss (CE 기반)
# ==================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: CE에 (1-pt)^gamma 가중치 적용.
    어려운 샘플(낮은 신뢰도)에 더 높은 가중치.
    """
    def __init__(self, gamma=2.0, weights=None):
        super().__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, logits, targets):
        if self.weights is not None:
            if self.weights.device != logits.device:
                self.weights = self.weights.to(logits.device)
            ce_loss = F.cross_entropy(logits, targets.long(), weight=self.weights, reduction='none')
        else:
            ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return ((1 - pt).pow(self.gamma) * ce_loss).mean()


# ==================================================================
# Logit Adjustment (Menon et al., ICLR 2021)
# ==================================================================
class LogitAdjustedLoss(nn.Module):
    """
    Logit Adjustment: logits + tau * log(prior) 로 보정 후 표준 CE.
    가중치가 아니라 logit shift로 불균형을 다루는 baseline (LDAM 대안).
    tau=0 이면 CE와 동일.
    """
    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        counts = np.array(class_counts, dtype=np.float32)
        prior = counts / np.sum(counts)
        self.adj = torch.tensor(tau * np.log(prior + 1e-12), dtype=torch.float32)

    def forward(self, logits, targets):
        if self.adj.device != logits.device:
            self.adj = self.adj.to(logits.device)
        return F.cross_entropy(logits + self.adj, targets.long())


# ==================================================================
# 분류용 마스터 Loss 클래스
# ==================================================================
class ClassificationLoss(nn.Module):
    """
    loss_name 문자열 파싱으로 손실함수 선택.

    지원하는 loss_name (11종):
        'ce'           → CrossEntropyLoss (가중치 없음)
        'wce'          → 가중 CE (역빈도)
        'pwce'         → 파워 가중 CE (alpha 파라미터)
        'sqce'         → 역제곱근 빈도 가중 CE (w ∝ 1/√n, 파라미터 없음)
        'lwce'         → 로그 가중 CE
        'plwce'        → 파워-로그 가중 CE (alpha 파라미터)
        'eslwce'       → eps 스무딩 로그 가중 CE (eps 파라미터)
        'combined'     → PLWCE+ES-LWCE 결합 (alpha, eps — orthogonality ablation)
        'cb'           → 클래스 균형 CE (beta 파라미터)
        'focal'        → CE 가중치 + Focal Loss (gamma 파라미터)
        'logitadj'     → Logit Adjustment (tau 파라미터, 가중치 아님)

    Args:
        loss_name: 위의 11가지 문자열 중 하나
        class_counts: 각 클래스의 샘플 수 리스트
        alpha: PLWCE / combined 지수 (기본값 1.0)
        beta: CB 베타 (기본값 0.9999)
        gamma: Focal gamma (기본값 2.0)
        eps: ES-LWCE / combined 스무딩 상수 (기본값 0.1)
        tau: Logit Adjustment 강도 (기본값 1.0)
    """
    def __init__(self, loss_name: str, class_counts, alpha=1.0, beta=0.9999, gamma=2.0,
                 eps=0.1, tau=1.0):
        super().__init__()
        self.loss_name = loss_name

        # 가중치 모드 결정 (파싱 우선순위)
        # 'eslwce'는 'lwce'를 부분문자열로 포함 → 반드시 'lwce'보다 먼저 검사
        if 'combined' in loss_name:
            weight_mode = 'combined'
        elif 'plwce' in loss_name:
            weight_mode = 'plwce'
        elif 'eslwce' in loss_name:
            weight_mode = 'eslwce'
        elif 'lwce' in loss_name:
            weight_mode = 'lwce'
        elif 'pwce' in loss_name:
            weight_mode = 'pwce'
        elif 'sqce' in loss_name:
            weight_mode = 'sqce'
        elif 'wce' in loss_name:
            weight_mode = 'wce'
        elif 'cb' in loss_name:
            weight_mode = 'cb'
        else:
            weight_mode = 'ce'

        self.weights = None
        if weight_mode != 'ce':
            if class_counts is None:
                raise ValueError(f"Loss '{loss_name}' requires class_counts.")
            self.weights = calculate_weights(class_counts, mode=weight_mode, alpha=alpha,
                                             beta=beta, eps=eps)

        # criterion 선택
        if 'logitadj' in loss_name:
            self.criterion = LogitAdjustedLoss(class_counts, tau=tau)
        elif 'focal' in loss_name:
            self.criterion = FocalLoss(gamma=gamma, weights=self.weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def get_weights(self):
        """실제 사용된 정규화 가중치 (없으면 None=균일). 논문 weight-distribution 분석용."""
        if self.weights is None:
            return None
        return self.weights.detach().cpu().numpy()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 디바이스 동기화 (Colab GPU 불일치 방지)
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            if self.criterion.weight is not None and \
               self.criterion.weight.device != logits.device:
                self.criterion.weight = self.criterion.weight.to(logits.device)
        return self.criterion(logits, targets.long())


def get_clf_loss(loss_name: str, class_counts, alpha=1.0, beta=0.9999, gamma=2.0,
                 eps=0.1, tau=1.0) -> ClassificationLoss:
    """Factory 함수 — 항상 이를 통해 손실함수를 생성하세요."""
    return ClassificationLoss(loss_name, class_counts, alpha, beta, gamma, eps, tau)
