"""
분류용 손실함수 — medical_data에서 필요한 함수만 추출.
calculate_weights()와 FocalLoss만 사용 (세그멘테이션 손실 제외).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================================================================
# 가중치 계산 함수
# ==================================================================
def calculate_weights(class_counts, mode='ce', alpha=1.0, beta=0.9999):
    """
    클래스 불균형을 보정하기 위한 가중치 계산.

    Args:
        class_counts: 각 클래스의 샘플 수 리스트
        mode: 'ce' (균일), 'wce' (역빈도), 'lwce' (로그 가중치),
              'plwce' (파워 로그 가중치), 'cb' (클래스 균형)
        alpha: plwce 지수
        beta: cb 베타 파라미터

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
    elif mode == 'lwce':
        weights = 1.0 / np.log1p(safe_counts)
    elif mode == 'plwce':
        weights = 1.0 / np.power(np.log1p(safe_counts), alpha)
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
# 분류용 마스터 Loss 클래스
# ==================================================================
class ClassificationLoss(nn.Module):
    """
    loss_name 문자열 파싱으로 손실함수 선택.

    지원하는 loss_name (6종):
        'ce'           → CrossEntropyLoss (가중치 없음)
        'wce'          → 가중 CE (역빈도)
        'lwce'         → 로그 가중 CE
        'plwce'        → 파워-로그 가중 CE (alpha 파라미터)
        'cb'           → 클래스 균형 CE (beta 파라미터)
        'plwce_focal'  → PLWCE 가중치 + Focal Loss (alpha, gamma 파라미터)

    Args:
        loss_name: 위의 6가지 문자열 중 하나
        class_counts: 각 클래스의 샘플 수 리스트
        alpha: PLWCE 지수 (기본값 1.0)
        beta: CB 베타 (기본값 0.9999)
        gamma: Focal gamma (기본값 2.0)
    """
    def __init__(self, loss_name: str, class_counts, alpha=1.0, beta=0.9999, gamma=2.0):
        super().__init__()
        self.loss_name = loss_name

        # 가중치 모드 결정 (파싱 우선순위)
        if 'plwce' in loss_name:
            weight_mode = 'plwce'
        elif 'lwce' in loss_name:
            weight_mode = 'lwce'
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
            self.weights = calculate_weights(class_counts, mode=weight_mode, alpha=alpha, beta=beta)

        # Focal Loss 또는 CrossEntropyLoss 선택
        if 'focal' in loss_name:
            self.criterion = FocalLoss(gamma=gamma, weights=self.weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 디바이스 동기화 (Colab GPU 불일치 방지)
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            if self.criterion.weight is not None and \
               self.criterion.weight.device != logits.device:
                self.criterion.weight = self.criterion.weight.to(logits.device)
        return self.criterion(logits, targets.long())


def get_clf_loss(loss_name: str, class_counts, alpha=1.0, beta=0.9999, gamma=2.0
                 ) -> ClassificationLoss:
    """Factory 함수 — 항상 이를 통해 손실함수를 생성하세요."""
    return ClassificationLoss(loss_name, class_counts, alpha, beta, gamma)
