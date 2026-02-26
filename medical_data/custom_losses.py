# src/custom_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================================================================
# 1. 가중치 계산 함수 (그대로 유지)
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
# 2. Dice & Focal Loss
# ==================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_background=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        true_1_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_1_hot, dim=dims)
        cardinality = torch.sum(probs + true_1_hot, dim=dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        if self.ignore_background:
            dice_score = dice_score[1:]
            
        return 1.0 - torch.mean(dice_score)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, logits, targets):
        # weights가 여기서 alpha 역할을 합니다.
        if self.weights is not None:
            if self.weights.device != logits.device:
                self.weights = self.weights.to(logits.device)
            ce_loss = F.cross_entropy(logits, targets.long(), weight=self.weights, reduction='none')
        else:
            ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = (1 - pt).pow(self.gamma) * ce_loss
        
        return focal_loss.mean()

# ==================================================================
# 3. 마스터 Loss 클래스 (로직 개선)
# ==================================================================
class SegmentationLoss(nn.Module):
    def __init__(self, loss_name='ce_dice', class_counts=None, alpha=1.0, beta=0.9999, gamma=2.0, 
                 lambda_dice=0.5, lambda_main=0.5):
        """
        문자열 파싱을 통해 [가중치 방식] + [Loss 종류]를 조합합니다.
        예: 'lwce_focal_dice' -> 가중치는 LWCE로 계산해서 Focal Loss에 넣고 Dice와 합침.
        """
        super(SegmentationLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_main = lambda_main
        
        # 1. 가중치 모드 파싱
        weight_mode = 'ce'
        if 'plwce' in loss_name: weight_mode = 'plwce'
        elif 'lwce' in loss_name: weight_mode = 'lwce'
        elif 'pwce' in loss_name: weight_mode = 'pwce'
        elif 'wce' in loss_name: weight_mode = 'wce'
        elif 'cb' in loss_name: weight_mode = 'cb'
        
        # 2. 가중치 계산
        self.weights = None
        if weight_mode != 'ce':
            if class_counts is None:
                raise ValueError(f"Loss '{loss_name}' requires class_counts.")
            self.weights = calculate_weights(class_counts, mode=weight_mode, alpha=alpha, beta=beta)
            print(f"[{loss_name}] Weights ({weight_mode}): Generated.")

        # 3. 메인 Loss 결정 (Focal vs CE)
        if 'focal' in loss_name:
            self.main_loss = FocalLoss(gamma=gamma, weights=self.weights)
        else:
            self.main_loss = nn.CrossEntropyLoss(weight=self.weights)

        # 4. Dice Loss
        self.dice_loss = DiceLoss(ignore_background=True)

    def forward(self, logits, targets):
        if isinstance(self.main_loss, nn.CrossEntropyLoss):
             if self.main_loss.weight is not None and self.main_loss.weight.device != logits.device:
                self.main_loss.weight = self.main_loss.weight.to(logits.device)
        
        loss_d = self.dice_loss(logits, targets)
        loss_m = self.main_loss(logits, targets.long())
        return self.lambda_dice * loss_d + self.lambda_main * loss_m

# ==================================================================
# 4. Factory 함수
# ==================================================================
def get_loss_function(loss_name, class_counts, alpha=1.0, beta=0.9999, gamma=2.0):
    return SegmentationLoss(loss_name, class_counts, alpha, beta, gamma)