# src/custom_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

# ==================================================================
# 1. 가중치 계산 함수
# ==================================================================
def calculate_weights(class_counts, mode='ce', alpha=1.0, beta=0.9999):
    counts = np.array(class_counts, dtype=np.float32)
    total = np.sum(counts)
    safe_counts = counts + 1e-6

    if mode == 'ce':
        weights = np.ones_like(counts)
    elif mode == 'wce':
        weights = total / safe_counts
    elif mode == 'pwce':
        weights = np.power(total / safe_counts, alpha)
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
# 2. 거리 맵 유틸리티
# ==================================================================
def compute_signed_distance_map(mask_np):
    """
    부호 있는 거리 맵 계산 (Kervadec et al., 2019 Boundary Loss 기준).
    - 마스크 내부: 음수 (foreground 예측을 장려)
    - 마스크 외부: 양수 (background 예측을 장려)
    mask_np: binary numpy array [H, W], dtype float32
    """
    mask = mask_np.astype(bool)
    if mask.any() and not mask.all():
        dist_in  = distance_transform_edt(mask).astype(np.float32)
        dist_out = distance_transform_edt(~mask).astype(np.float32)
    elif not mask.any():          # 전부 배경
        dist_in  = np.zeros_like(mask_np, dtype=np.float32)
        dist_out = distance_transform_edt(np.ones_like(mask_np)).astype(np.float32)
    else:                         # 전부 병변
        dist_in  = distance_transform_edt(np.ones_like(mask_np)).astype(np.float32)
        dist_out = np.zeros_like(mask_np, dtype=np.float32)
    return dist_out - dist_in     # 외부 양수, 내부 음수


# ==================================================================
# 3. Region-based Losses (Dice / Tversky / Focal Tversky)
# ==================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_background=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        true_1_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_1_hot, dim=dims)
        cardinality  = torch.sum(probs + true_1_hot, dim=dims)
        dice_score   = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        if self.ignore_background:
            dice_score = dice_score[1:]
        return 1.0 - torch.mean(dice_score)


class TverskyLoss(nn.Module):
    """
    Tversky = TP / (TP + alpha*FN + beta*FP)
    alpha=0.7, beta=0.3 → FN 패널티 강화 (소형 병변 Recall 향상)
    alpha=beta=0.5 → Dice와 동일
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5, ignore_background=True):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        true_1_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        TP = torch.sum(probs * true_1_hot, dim=dims)
        FP = torch.sum(probs * (1 - true_1_hot), dim=dims)
        FN = torch.sum((1 - probs) * true_1_hot, dim=dims)
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        if self.ignore_background:
            tversky = tversky[1:]
        return 1.0 - torch.mean(tversky)


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky = (1 - Tversky)^gamma
    gamma < 1 → 어려운 샘플(소형 병변) 집중 (Abraham & Khan, 2019)
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5, ignore_background=True):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        true_1_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        TP = torch.sum(probs * true_1_hot, dim=dims)
        FP = torch.sum(probs * (1 - true_1_hot), dim=dims)
        FN = torch.sum((1 - probs) * true_1_hot, dim=dims)
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        if self.ignore_background:
            tversky = tversky[1:]
        return torch.mean((1.0 - tversky).pow(self.gamma))


# ==================================================================
# 4. Boundary-based Losses (BL / LBL)
# ==================================================================
class BoundaryLoss(nn.Module):
    """
    Boundary Loss (Kervadec et al., 2019).
    부호 있는 거리 맵과 foreground 예측 확률의 내적.
    단독 사용 불가 — 반드시 Dice 등 region-based loss와 결합해야 함.
    """
    def _get_dist_map(self, targets):
        B = targets.shape[0]
        maps = []
        for i in range(B):
            mask = (targets[i] > 0).cpu().numpy().astype(np.float32)
            maps.append(compute_signed_distance_map(mask))
        dist = torch.from_numpy(np.stack(maps)).float().to(targets.device)
        # [-1, 1] 정규화 (이미지 크기 의존성 제거)
        scale = dist.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        return dist / scale

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        # binary: class 1 확률, multi-class: 전체 foreground 확률
        fg_prob = probs[:, 1] if logits.shape[1] == 2 else 1.0 - probs[:, 0]
        dist_map = self._get_dist_map(targets)
        return (fg_prob * dist_map).mean()


class LogBoundaryLoss(nn.Module):
    """
    Log-Boundary Loss (신규 제안).
    BL의 거리 맵에 log(1 + |d|)를 적용해 먼 배경 픽셀의 과대 가중치를 억제.
    LWCE와 동일한 log-scaling 철학을 픽셀 거리 도메인에 확장.

    BL 대비 차이:
      BL  weight ∝ |d|          → 배경 중심부 픽셀이 경계 근처보다 훨씬 큰 가중치
      LBL weight ∝ log(1+|d|)   → 거리 범위를 압축, 쉬운(명백한 배경) 픽셀 비중 감소
    """
    def _get_log_dist_map(self, targets):
        B = targets.shape[0]
        maps = []
        for i in range(B):
            mask = (targets[i] > 0).cpu().numpy().astype(np.float32)
            signed = compute_signed_distance_map(mask)
            log_dist = np.sign(signed) * np.log1p(np.abs(signed))
            maps.append(log_dist.astype(np.float32))
        dist = torch.from_numpy(np.stack(maps)).float().to(targets.device)
        scale = dist.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        return dist / scale

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        fg_prob = probs[:, 1] if logits.shape[1] == 2 else 1.0 - probs[:, 0]
        dist_map = self._get_log_dist_map(targets)
        return (fg_prob * dist_map).mean()


# ==================================================================
# 5. Focal Loss (CE 기반)
# ==================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weights=None):
        super().__init__()
        self.gamma   = gamma
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
# 6. 마스터 Loss 클래스
# ==================================================================
class SegmentationLoss(nn.Module):
    """
    loss_name 문자열 파싱으로 [CE 가중치] + [Region Loss] + [Boundary Loss] 조합.

    지원 이름 예시:
      ce_dice                  → CE + Dice  (λ=0.5/0.5)
      lwce_dice                → LWCE-CE + Dice
      plwce_dice               → PLWCE-CE + Dice
      plwce_focal_dice         → PLWCE-FocalCE + Dice
      ce_tversky               → CE + Tversky
      plwce_focal_tversky      → PLWCE-CE + FocalTversky
      ce_dice_boundary         → CE + Dice + BL  (λ=1/3 each)
      plwce_dice_boundary      → PLWCE-CE + Dice + BL
      ce_dice_log_boundary     → CE + Dice + LBL
      plwce_dice_log_boundary  → PLWCE-CE + Dice + LBL
      plwce_boundary           → PLWCE-CE + BL  (λ=0.5/0.5, Dice 없음)
      plwce_log_boundary       → PLWCE-CE + LBL (λ=0.5/0.5, Dice 없음)

    lambda는 활성 컴포넌트 수에 따라 자동 균등 분배.
    """
    def __init__(self, loss_name='ce_dice', class_counts=None, alpha=1.0, beta=0.9999, gamma=2.0):
        super().__init__()

        # --- 1. CE 가중치 모드 파싱 ---
        weight_mode = 'ce'
        if   'plwce' in loss_name: weight_mode = 'plwce'
        elif 'lwce'  in loss_name: weight_mode = 'lwce'
        elif 'pwce'  in loss_name: weight_mode = 'pwce'
        elif 'wce'   in loss_name: weight_mode = 'wce'
        elif 'cb'    in loss_name: weight_mode = 'cb'

        self.weights = None
        if weight_mode != 'ce':
            if class_counts is None:
                raise ValueError(f"Loss '{loss_name}' requires class_counts.")
            self.weights = calculate_weights(class_counts, mode=weight_mode, alpha=alpha, beta=beta)
            print(f"[{loss_name}] CE weights ({weight_mode}): Generated.")

        # --- 2. 메인 CE Loss ---
        if 'focal' in loss_name and 'focal_tversky' not in loss_name:
            self.ce_loss = FocalLoss(gamma=gamma, weights=self.weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)

        # --- 3. Region Loss (Dice / Tversky / FocalTversky / 없음) ---
        if 'focal_tversky' in loss_name:
            self.region_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=gamma)
        elif 'tversky' in loss_name:
            self.region_loss = TverskyLoss(alpha=0.7, beta=0.3)
        elif 'dice' in loss_name:
            self.region_loss = DiceLoss(ignore_background=True)
        else:
            self.region_loss = None

        # --- 4. Boundary Loss (LBL / BL / 없음) ---
        # log_boundary 먼저 확인 (boundary의 substring 충돌 방지)
        if 'log_boundary' in loss_name:
            self.boundary_loss = LogBoundaryLoss()
        elif 'boundary' in loss_name:
            self.boundary_loss = BoundaryLoss()
        else:
            self.boundary_loss = None

        # --- 5. Lambda: 활성 컴포넌트 수로 균등 분배 (초기값) ---
        n = 1 + (self.region_loss is not None) + (self.boundary_loss is not None)
        lam = 1.0 / n
        self.lam_ce       = lam
        self.lam_region   = lam if self.region_loss   is not None else 0.0
        self.lam_boundary = lam if self.boundary_loss is not None else 0.0
        print(f"[{loss_name}] Components: CE"
              + (f" + {type(self.region_loss).__name__}" if self.region_loss else "")
              + (f" + {type(self.boundary_loss).__name__}" if self.boundary_loss else "")
              + f"  (λ={lam:.3f} each)")

    def set_boundary_alpha(self, alpha_t):
        """Boundary Loss annealing: BL 비중을 alpha_t로, 나머지를 균등 분배.
        alpha_t = min(epoch / T_max, 0.5) 형태로 호출.
        boundary_loss가 없으면 무시.
        """
        if self.boundary_loss is None:
            return
        n_non_boundary = 1 + (self.region_loss is not None)
        self.lam_boundary = alpha_t
        self.lam_ce       = (1.0 - alpha_t) / n_non_boundary
        self.lam_region   = (1.0 - alpha_t) / n_non_boundary if self.region_loss is not None else 0.0

    def forward(self, logits, targets):
        if isinstance(self.ce_loss, nn.CrossEntropyLoss):
            if self.ce_loss.weight is not None and self.ce_loss.weight.device != logits.device:
                self.ce_loss.weight = self.ce_loss.weight.to(logits.device)

        loss = self.lam_ce * self.ce_loss(logits, targets.long())
        if self.region_loss is not None:
            loss = loss + self.lam_region * self.region_loss(logits, targets)
        if self.boundary_loss is not None:
            loss = loss + self.lam_boundary * self.boundary_loss(logits, targets)
        return loss


# ==================================================================
# 7. Factory 함수
# ==================================================================
def get_loss_function(loss_name, class_counts, alpha=1.0, beta=0.9999, gamma=2.0):
    return SegmentationLoss(loss_name, class_counts, alpha, beta, gamma)
