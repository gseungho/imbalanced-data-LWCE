# src/custom_losses.py

import numpy as np # 수치 연산을 위한 모듈
from scipy.special import expit, softmax # 시그모이드(Sigmoid)와 소프트맥스(Softmax) 함수

# --- Helper Function ---
def _get_class_counts(y_true, n_classes): # 클래스별 샘플 수를 계산하는 보조 함수
    return np.bincount(y_true.astype(int), minlength=n_classes) # minlength: y_true에 없는 클래스도 0으로 포함하여 반환

# --- 이진 분류용 손실 함수들 ---
class WeightedCrossEntropy: # 클래스 빈도의 역수를 가중치로 사용하는 손실 함수
    def __init__(self, weighting_scheme: str = 'inverse_frequency'): # 생성자: 가중치 계산 방식('역수' 또는 '역수 제곱근') 설정
        self.weighting_scheme = weighting_scheme; self.class_weight_values = None
    def initialize(self, y_true: np.ndarray): # 학습 시작 전 클래스별 가중치를 미리 계산하는 메서드
        n0 = np.sum(1 - y_true); n1 = np.sum(y_true) # 0 클래스와 1 클래스의 샘플 수 계산
        if self.weighting_scheme == 'inverse_frequency': # '역수' 방식
            w0 = (n0 + n1) / (2 * n0) if n0 > 0 else 1; w1 = (n0 + n1) / (2 * n1) if n1 > 0 else 1 # 가중치 계산
        elif self.weighting_scheme == 'inverse_sqrt_frequency': # '역수 제곱근' 방식
            w0 = np.sqrt((n0 + n1) / n0) if n0 > 0 else 1; w1 = np.sqrt((n0 + n1) / n1) if n1 > 0 else 1 # 가중치 계산
        else: raise ValueError("지원되지 않는 가중치 방식입니다.")
        self.class_weight_values = {0: w0, 1: w1} # 계산된 가중치를 딕셔너리로 저장
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사(gradient)와 헤시안(Hessian)을 계산
        y_true = np.array(y_true) # y_true를 numpy 배열로 변환
        weights = np.where(y_true == 1, self.class_weight_values[1], self.class_weight_values[0]) # 각 샘플의 클래스에 맞는 가중치 할당
        p = expit(y_pred) # 모델의 예측 점수(logit)를 시그모이드 함수로 0~1 사이 확률로 변환
        grad = weights * (p - y_true) # 가중치가 적용된 경사 계산
        hess = weights * p * (1 - p) # 가중치가 적용된 헤시안 계산
        if sample_weight is not None: grad *= sample_weight; hess *= sample_weight # 외부에서 주어진 샘플 가중치 추가 적용
        return grad, hess # 최종 경사와 헤시안 반환
    
# --- [신규 추가] Power Scaled Inverse (이진 분류) ---
class PowerScaledInverseCE:
    """
    Inverse Frequency의 Power Scaled 버전
    w_i = (N / (C * n_i))^α
    """
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: power scaling 파라미터 (0.5 = sqrt, 1.0 = inverse)
        """
        self.alpha = alpha
        self.class_weight_values = None
    
    def initialize(self, y_true: np.ndarray):
        n0 = np.sum(1 - y_true)
        n1 = np.sum(y_true)
        total = n0 + n1
        
        # Power scaled inverse frequency
        w0 = np.power(total / (2 * n0 + 1e-9), self.alpha) if n0 > 0 else 1
        w1 = np.power(total / (2 * n1 + 1e-9), self.alpha) if n1 > 0 else 1
        
        self.class_weight_values = {0: w0, 1: w1}
    
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple:
        y_true = np.array(y_true)
        weights = np.where(y_true == 1, self.class_weight_values[1], self.class_weight_values[0])
        p = expit(y_pred)
        grad = weights * (p - y_true)
        hess = weights * p * (1 - p)
        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight
        return grad, hess


class FocalLoss: # 분류하기 어려운 샘플에 더 집중하는 손실 함수
    def __init__(self, alpha=0.25, gamma=2.0): # 생성자: alpha는 클래스 가중치, gamma는 집중도 조절 파라미터
        self.alpha = alpha; self.gamma = gamma
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        y_true = np.array(y_true); p = expit(y_pred) # 예측 점수를 확률로 변환
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha) # 정답 클래스에 맞는 alpha 가중치 선택
        p_t = np.where(y_true == 1, p, 1 - p) # 정답 클래스에 대한 예측 확률
        grad = alpha_t * np.power(1 - p_t, self.gamma) * (p - y_true) # Focal Loss의 경사 공식
        epsilon = 1e-9; hess_term = (1 + self.gamma / (1 - p_t + epsilon) * (y_true - p)) # 0으로 나누기 방지 및 헤시안 계산 항
        hess = alpha_t * np.power(1 - p_t, self.gamma) * p * (1 - p) * hess_term # Focal Loss의 헤시안 공식
        if sample_weight is not None: grad *= sample_weight; hess *= sample_weight # 외부 샘플 가중치 적용
        return grad, hess # 최종 경사와 헤시안 반환

class ClassBalancedLoss: # "Effective Number of Samples" 개념 기반 손실 함수
    def __init__(self, base_loss_func, beta: float = 0.999): # 생성자: 기반 손실 함수와 beta 파라미터 설정
        self.base_loss_func = base_loss_func; self.beta = beta; self.class_weight_values = None
    def initialize(self, y_true: np.ndarray): # 학습 전 가중치 미리 계산
        samples_per_class = np.array([np.sum(1 - y_true), np.sum(y_true)]) # 클래스별 샘플 수
        effective_num = (1.0 - np.power(self.beta, samples_per_class)) / (1.0 - self.beta) # 유효 샘플 수 계산
        weights = 1.0 / effective_num; normalized_weights = weights / np.sum(weights) * len(samples_per_class) # 가중치 계산 및 정규화
        self.class_weight_values = {0: normalized_weights[0], 1: normalized_weights[1]} # 가중치 저장
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        base_grad, base_hess = self.base_loss_func(y_true, y_pred) # 기반 손실 함수의 경사/헤시안 먼저 계산
        y_true = np.array(y_true); weights = np.where(y_true == 1, self.class_weight_values[1], self.class_weight_values[0]) # CB 가중치 할당
        grad = base_grad * weights; hess = base_hess * weights # 계산된 가중치를 경사/헤시안에 곱해줌
        if sample_weight is not None: grad *= sample_weight; hess *= sample_weight # 외부 샘플 가중치 적용
        return grad, hess

class AdaptiveWeightedCrossEntropy: # 클래스 빈도의 로그 값에 반비례하는 가중치를 사용하는 손실 함수
    def __init__(self): self.class_weight_values = None # 생성자
    def initialize(self, y_true: np.ndarray): # 학습 전 가중치 미리 계산
        n0 = np.sum(1 - y_true); n1 = np.sum(y_true) # 클래스별 샘플 수
        w0_raw = 1 / np.log(1 + n0) if n0 > 0 else 1; w1_raw = 1 / np.log(1 + n1) if n1 > 0 else 1 # 로그 기반 가중치 계산
        total_w = w0_raw + w1_raw; w0 = w0_raw / total_w * 2 if total_w > 0 else 1; w1 = w1_raw / total_w * 2 if total_w > 0 else 1 # 가중치 정규화
        self.class_weight_values = {0: w0, 1: w1} # 가중치 저장
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        y_true = np.array(y_true); weights = np.where(y_true == 1, self.class_weight_values[1], self.class_weight_values[0]) # 가중치 할당
        p = expit(y_pred); grad = weights * (p - y_true); hess = weights * p * (1 - p) # 가중치 적용 경사/헤시안 계산
        if sample_weight is not None: grad *= sample_weight; hess *= sample_weight # 외부 샘플 가중치 적용
        return grad, hess

# --- [신규 추가] Power Scaled Adaptive CE (이진 분류) ---
class PowerScaledAdaptiveCE: # Adaptive CE의 Power Scaled 버전 (PDF 23페이지)
    def __init__(self, alpha: float = 1.0): # 생성자, alpha: 파워 스케일링 파라미터
        self.alpha = alpha; self.class_weight_values = None
    def initialize(self, y_true: np.ndarray): # 학습 전 가중치 미리 계산
        n0 = np.sum(1 - y_true); n1 = np.sum(y_true) # 클래스별 샘플 수
        # 1e-9를 더해 log(1) = 0이 되어 0으로 나누는 오류 방지
        w0_raw = 1 / np.power(np.log(1 + n0 + 1e-9), self.alpha) if n0 > 0 else 1
        w1_raw = 1 / np.power(np.log(1 + n1 + 1e-9), self.alpha) if n1 > 0 else 1
        total_w = w0_raw + w1_raw; w0 = w0_raw / total_w * 2 if total_w > 0 else 1; w1 = w1_raw / total_w * 2 if total_w > 0 else 1 # 가중치 정규화
        self.class_weight_values = {0: w0, 1: w1} # 가중치 저장
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        y_true = np.array(y_true); weights = np.where(y_true == 1, self.class_weight_values[1], self.class_weight_values[0]) # 가중치 할당
        p = expit(y_pred); grad = weights * (p - y_true); hess = weights * p * (1 - p) # 가중치 적용 경사/헤시안 계산
        if sample_weight is not None: grad *= sample_weight; hess *= sample_weight # 외부 샘플 가중치 적용
        return grad, hess
# --- [신규 추가] ---

# --- [신규 추가] Rho Power Scaled Inverse (이진 분류) ---
class RhoPowerScaledInverseCE:
    """
    Power Scaled Inverse CE + Rho 파라미터 (이진 분류)
    소수 클래스(레이블 1)에 rho를 곱하여 추가 부스팅
    """
    def __init__(self, alpha: float = 0.5, rho: float = 1.0):
        self.alpha = alpha
        self.rho = rho
        self.class_weight_values = None
    
    def initialize(self, y_true: np.ndarray):
        n0 = np.sum(1 - y_true)
        n1 = np.sum(y_true)
        total = n0 + n1
        
        w0 = np.power(total / (2 * n0 + 1e-9), self.alpha) if n0 > 0 else 1
        w1 = np.power(total / (2 * n1 + 1e-9), self.alpha) if n1 > 0 else 1
        
        # 소수 클래스(레이블 1)에 rho 적용
        self.class_weight_values = {0: w0, 1: w1 * self.rho}
    
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple:
        y_true = np.array(y_true)
        weights = np.where(y_true == 1, self.class_weight_values[1], self.class_weight_values[0])
        p = expit(y_pred)
        grad = weights * (p - y_true)
        hess = weights * p * (1 - p)
        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight
        return grad, hess

# --- [신규 추가] Rho Power Scaled Adaptive CE (이진 분류) ---
class RhoPowerScaledAdaptiveCE:
    """
    Power Scaled Adaptive CE + Rho 파라미터 (이진 분류)
    소수 클래스(레이블 1)에 rho를 곱하여 추가 부스팅
    """
    def __init__(self, alpha: float = 1.0, rho: float = 1.0):
        self.alpha = alpha
        self.rho = rho
        self.class_weight_values = None
        
    def initialize(self, y_true: np.ndarray):
        n0 = np.sum(1 - y_true); n1 = np.sum(y_true)
        w0_raw = 1 / np.power(np.log(1 + n0 + 1e-9), self.alpha) if n0 > 0 else 1
        w1_raw = 1 / np.power(np.log(1 + n1 + 1e-9), self.alpha) if n1 > 0 else 1
        
        # 정규화 전 소수 클래스에 rho 적용
        w1_raw *= self.rho
        
        total_w = w0_raw + w1_raw
        w0 = w0_raw / total_w * 2 if total_w > 0 else 1
        w1 = w1_raw / total_w * 2 if total_w > 0 else 1
        self.class_weight_values = {0: w0, 1: w1}

    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple:
        y_true = np.array(y_true)
        weights = np.where(y_true == 1, self.class_weight_values[1], self.class_weight_values[0])
        p = expit(y_pred)
        grad = weights * (p - y_true)
        hess = weights * p * (1 - p)
        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight
        return grad, hess

class ClassBalancedFocalLoss(ClassBalancedLoss): # ClassBalancedLoss와 FocalLoss를 결합한 손실 함수
    def __init__(self, beta=0.999, gamma=2.0): # 생성자
        base_focal_loss = lambda yt, yp: FocalLoss(alpha=1.0, gamma=gamma).compute_grad_hess(yt, yp) # 기반 함수로 FocalLoss(alpha=1) 사용
        super().__init__(base_loss_func=base_focal_loss, beta=beta) # 부모 클래스(ClassBalancedLoss) 초기화

# --- 다중 클래스용 손실 함수들 ---

class MultiClassWeightedCrossEntropy: # 다중 클래스용 WeightedCrossEntropy
    def __init__(self, n_classes: int, weighting_scheme: str = 'inverse_frequency', model_type: str = 'xgboost'): # 생성자
        self.n_classes = n_classes; self.weighting_scheme = weighting_scheme; self.model_type = model_type; self.class_weights = None
    def initialize(self, y_true: np.ndarray): # 학습 전 가중치 미리 계산
        class_counts = _get_class_counts(y_true, self.n_classes); num_classes_from_data = len(class_counts); total_samples = np.sum(class_counts)
        if self.weighting_scheme == 'inverse_frequency': weights = total_samples / (num_classes_from_data * class_counts + 1e-9)
        elif self.weighting_scheme == 'inverse_sqrt_frequency': weights = np.sqrt(total_samples / (num_classes_from_data * class_counts + 1e-9))
        else: raise ValueError("지원되지 않는 가중치 방식입니다.")
        self.class_weights = weights
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        y_true = np.atleast_1d(y_true).astype(int); num_classes_actual = len(self.class_weights) # 실제 클래스 개수 확인
        y_pred_reshaped = y_pred.reshape(len(y_true), num_classes_actual) # 1D 예측값을 (샘플 수, 클래스 수) 2D 배열로 변환
        prob = softmax(y_pred_reshaped, axis=1) # 모델 점수를 소프트맥스 함수로 다중 클래스 확률로 변환
        y_true_one_hot = np.eye(num_classes_actual)[y_true] # 실제 정답을 원-핫 벡터로 변환 (예: 2 -> [0,0,1,0...])
        grad = prob - y_true_one_hot # 다중 클래스 Cross-Entropy의 경사 계산
        if self.model_type == 'lightgbm': hess = prob # LightGBM 실험을 위한 안정화된 헤시안
        else: hess = prob * (1.0 - prob) # XGBoost용 표준 근사 헤시안
        sample_weights_internal = self.class_weights[y_true] # 클래스별 내부 가중치 적용
        grad = grad * sample_weights_internal[:, np.newaxis]; hess = hess * sample_weights_internal[:, np.newaxis] # 브로드캐스팅하여 적용
        if sample_weight is not None: grad = grad * sample_weight[:, np.newaxis]; hess = hess * sample_weight[:, np.newaxis] # 외부 샘플 가중치 적용
        return grad.flatten(), hess.flatten() # XGBoost API가 요구하는 1D 배열 형태로 변환하여 반환
    
# --- [신규 추가] Power Scaled Inverse (다중 클래스) ---
class MultiClassPowerScaledInverseCE:
    """
    다중 클래스용 Power Scaled Inverse CE
    """
    def __init__(self, n_classes: int, alpha: float = 0.5):
        self.n_classes = n_classes
        self.alpha = alpha
        self.class_weights = None
    
    def initialize(self, y_true: np.ndarray):
        class_counts = _get_class_counts(y_true, self.n_classes)
        num_classes_from_data = len(class_counts)
        total_samples = np.sum(class_counts)
        
        # Power scaled inverse frequency
        weights = np.power(
            total_samples / (num_classes_from_data * class_counts + 1e-9),
            self.alpha
        )
        
        self.class_weights = weights
    
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple:
        y_true = np.atleast_1d(y_true).astype(int)
        num_classes_actual = len(self.class_weights)
        y_pred_reshaped = y_pred.reshape(len(y_true), num_classes_actual)
        prob = softmax(y_pred_reshaped, axis=1)
        y_true_one_hot = np.eye(num_classes_actual)[y_true]
        
        grad = prob - y_true_one_hot
        hess = prob * (1.0 - prob)  # XGBoost용 표준 헤시안
        
        sample_weights_internal = self.class_weights[y_true]
        grad = grad * sample_weights_internal[:, np.newaxis]
        hess = hess * sample_weights_internal[:, np.newaxis]
        
        if sample_weight is not None:
            grad = grad * sample_weight[:, np.newaxis]
            hess = hess * sample_weight[:, np.newaxis]
        
        return grad.flatten(), hess.flatten()

class MultiClassAdaptiveCrossEntropy: # 다중 클래스용 AdaptiveCrossEntropy
    def __init__(self, n_classes: int, model_type: str = 'xgboost'): # 생성자
        self.n_classes = n_classes; self.model_type = model_type; self.class_weights = None
    def initialize(self, y_true: np.ndarray): # 학습 전 가중치 미리 계산
        class_counts = _get_class_counts(y_true, self.n_classes); num_classes_from_data = len(class_counts)
        raw_weights = 1.0 / np.log(1 + class_counts + 1e-9) # 로그 기반 가중치 계산
        self.class_weights = raw_weights / np.sum(raw_weights) * num_classes_from_data # 가중치 정규화
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        y_true = np.atleast_1d(y_true).astype(int); num_classes_actual = len(self.class_weights)
        y_pred_reshaped = y_pred.reshape(len(y_true), num_classes_actual); prob = softmax(y_pred_reshaped, axis=1)
        y_true_one_hot = np.eye(num_classes_actual)[y_true]
        grad = prob - y_true_one_hot
        if self.model_type == 'lightgbm': hess = prob
        else: hess = prob * (1.0 - prob)
        sample_weights_internal = self.class_weights[y_true]
        grad = grad * sample_weights_internal[:, np.newaxis]; hess = hess * sample_weights_internal[:, np.newaxis]
        if sample_weight is not None: grad = grad * sample_weight[:, np.newaxis]; hess = hess * sample_weight[:, np.newaxis]
        return grad.flatten(), hess.flatten()

# --- [신규 추가] Power Scaled Adaptive CE (다중 클래스) ---
class MultiClassPowerScaledAdaptiveCE: # 다중 클래스용 Power Scaled Adaptive CE (PDF 23페이지)
    def __init__(self, n_classes: int, alpha: float = 1.0, model_type: str = 'xgboost'): # 생성자, alpha: 파워 스케일링 파라미터
        self.n_classes = n_classes; self.alpha = alpha; self.model_type = model_type; self.class_weights = None
    def initialize(self, y_true: np.ndarray): # 학습 전 가중치 미리 계산
        class_counts = _get_class_counts(y_true, self.n_classes); num_classes_from_data = len(class_counts)
        # 1e-9를 더해 log(1) = 0이 되어 0으로 나누는 오류 방지
        raw_weights = 1.0 / np.power(np.log(1 + class_counts + 1e-9), self.alpha) # Power Scaled 로그 기반 가중치 계산
        self.class_weights = raw_weights / np.sum(raw_weights) * num_classes_from_data # 가중치 정규화
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        y_true = np.atleast_1d(y_true).astype(int); num_classes_actual = len(self.class_weights)
        y_pred_reshaped = y_pred.reshape(len(y_true), num_classes_actual); prob = softmax(y_pred_reshaped, axis=1)
        y_true_one_hot = np.eye(num_classes_actual)[y_true]
        grad = prob - y_true_one_hot
        if self.model_type == 'lightgbm': hess = prob
        else: hess = prob * (1.0 - prob)
        sample_weights_internal = self.class_weights[y_true]
        grad = grad * sample_weights_internal[:, np.newaxis]; hess = hess * sample_weights_internal[:, np.newaxis]
        if sample_weight is not None: grad = grad * sample_weight[:, np.newaxis]; hess = hess * sample_weight[:, np.newaxis]
        return grad.flatten(), hess.flatten()
# --- [신규 추가] ---

class MultiClassFocalLoss: # 다중 클래스용 FocalLoss
    def __init__(self, n_classes: int, alpha=0.25, gamma=2.0, model_type: str = 'xgboost'): # 생성자
        self.n_classes = n_classes; self.alpha = alpha; self.gamma = gamma; self.model_type = model_type
    def initialize(self, y_true: np.ndarray): # 학습 전 alpha 가중치 계산
        if isinstance(self.alpha, float): # alpha가 단일 값일 경우
            counts = _get_class_counts(y_true, self.n_classes); num_classes_from_data = len(counts); total_samples = float(sum(counts))
            class_weights = [total_samples / (num_classes_from_data * count + 1e-9) for count in counts]
            alpha_values = [(1 - self.alpha) if i == np.argmax(counts) else self.alpha for i in range(num_classes_from_data)]
            self.alpha = np.array(alpha_values) * class_weights # 클래스별 최종 alpha 가중치 계산
        else: self.alpha = np.array(self.alpha) # alpha가 배열로 주어지면 그대로 사용
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        y_true = np.atleast_1d(y_true).astype(int); num_classes_actual = len(self.alpha)
        y_pred_reshaped = y_pred.reshape(len(y_true), num_classes_actual); prob = softmax(y_pred_reshaped, axis=1)
        y_true_one_hot = np.eye(num_classes_actual)[y_true]
        p_t = np.sum(y_true_one_hot * prob, axis=1); alpha_t = self.alpha[y_true] # 정답 클래스에 대한 확률 및 alpha 가중치
        grad_common = (prob - y_true_one_hot).T; grad_weight = alpha_t * np.power(1 - p_t, self.gamma); grad = (grad_common * grad_weight).T # Focal Loss 경사
        if self.model_type == 'lightgbm': # LightGBM일 경우
            hess_weight = alpha_t * np.power(1 - p_t, self.gamma); hess = (prob.T * hess_weight).T # 안정화된 헤시안
        else: # XGBoost일 경우
            hess_common = (prob * (1 - prob)).T; hess_weight = alpha_t * np.power(1 - p_t, self.gamma) * (1 + self.gamma * p_t / (1 - p_t + 1e-9)); hess = (hess_common * hess_weight).T # 수학적으로 더 정확한 헤시안
        if sample_weight is not None: grad = grad * sample_weight[:, np.newaxis]; hess = hess * sample_weight[:, np.newaxis] # 외부 샘플 가중치 적용
        return grad.flatten(), hess.flatten() # 1D 배열로 변환하여 반환

class MultiClassClassBalancedLoss: # 다중 클래스용 ClassBalancedLoss
    def __init__(self, n_classes, base_loss_func, beta: float = 0.999): # 생성자
        self.n_classes = n_classes; self.base_loss_func = base_loss_func; self.beta = beta; self.class_weights = None
    def initialize(self, y_true: np.ndarray): # 학습 전 가중치 미리 계산
        samples_per_class = _get_class_counts(y_true, self.n_classes); num_classes_from_data = len(samples_per_class)
        effective_num = (1.0 - np.power(self.beta, samples_per_class)) / (1.0 - self.beta); weights = 1.0 / (effective_num + 1e-9) # 유효 샘플 수 기반 가중치 계산
        self.class_weights = weights / np.sum(weights) * num_classes_from_data # 가중치 정규화
        if hasattr(self.base_loss_func, 'initialize'): self.base_loss_func.initialize(y_true) # 기반 함수에 initialize가 있으면 호출
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        base_grad_flat, base_hess_flat = self.base_loss_func(y_true, y_pred) # 외부에서 전달받은 기반 함수의 경사/헤시안 계산
        y_true = np.atleast_1d(y_true).astype(int); num_classes_actual = len(self.class_weights)
        sample_weights_internal = self.class_weights[y_true] # CB 가중치 할당
        grad = base_grad_flat.reshape(len(y_true), num_classes_actual) * sample_weights_internal[:, np.newaxis] # 가중치 적용
        hess = base_hess_flat.reshape(len(y_true), num_classes_actual) * sample_weights_internal[:, np.newaxis] # 가중치 적용
        if sample_weight is not None: grad = grad * sample_weight[:, np.newaxis]; hess = hess * sample_weight[:, np.newaxis] # 외부 샘플 가중치 적용
        return grad.flatten(), hess.flatten() # 1D 배열로 변환하여 반환

class MultiClassClassBalancedFocalLoss: # 다중 클래스용 ClassBalancedFocalLoss
    def __init__(self, n_classes, beta=0.999, gamma=2.0, model_type: str = 'xgboost'): # 생성자
        self.n_classes = n_classes; self.beta = beta
        self.base_focal_loss = MultiClassFocalLoss(n_classes=n_classes, alpha=1.0, gamma=gamma, model_type=model_type) # 기반 함수로 MultiClassFocalLoss 사용
        self.cb_weights = None
    def initialize(self, y_true: np.ndarray): # 학습 전 가중치 미리 계산
        self.base_focal_loss.initialize(y_true) # 기반 함수의 initialize 호출
        samples_per_class = _get_class_counts(y_true, self.n_classes); num_classes_from_data = len(samples_per_class)
        effective_num = (1.0 - np.power(self.beta, samples_per_class)) / (1.0 - self.beta); weights = 1.0 / (effective_num + 1e-9) # CB 가중치 계산
        self.cb_weights = weights / np.sum(weights) * num_classes_from_data # 가중치 정규화
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple: # 경사/헤시안 계산
        base_grad_flat, base_hess_flat = self.base_focal_loss.compute_grad_hess(y_true, y_pred, sample_weight=None) # 기반 함수의 경사/헤시안 계산
        y_true = np.atleast_1d(y_true).astype(int); num_classes_actual = len(self.cb_weights)
        sample_weights_internal = self.cb_weights[y_true] # CB 가중치 할당
        grad = base_grad_flat.reshape(len(y_true), num_classes_actual) * sample_weights_internal[:, np.newaxis] # 가중치 적용
        hess = base_hess_flat.reshape(len(y_true), num_classes_actual) * sample_weights_internal[:, np.newaxis] # 가중치 적용
        if sample_weight is not None: grad = grad * sample_weight[:, np.newaxis]; hess = hess * sample_weight[:, np.newaxis] # 외부 샘플 가중치 적용

# --- [신규 추가] Rho MultiClass Power Scaled Inverse ---
class RhoMultiClassPowerScaledInverseCE:
    """
    다중 클래스용 Power Scaled Inverse CE + Rho 파라미터
    최다 빈도 클래스를 제외한 나머지 모든 클래스에 rho를 곱함
    """
    def __init__(self, n_classes: int, alpha: float = 0.5, rho: float = 1.0):
        self.n_classes = n_classes
        self.alpha = alpha
        self.rho = rho
        self.class_weights = None
    
    def initialize(self, y_true: np.ndarray):
        class_counts = _get_class_counts(y_true, self.n_classes)
        num_classes_from_data = len(class_counts)
        total_samples = np.sum(class_counts)
        
        # 기본 가중치 계산
        weights = np.power(
            total_samples / (num_classes_from_data * class_counts + 1e-9),
            self.alpha
        )
        
        # 다수 클래스 찾기 및 나머지 클래스에 rho 적용
        majority_class_idx = np.argmax(class_counts)
        for i in range(len(weights)):
            if i != majority_class_idx:
                 weights[i] *= self.rho

        self.class_weights = weights
    
    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple:
        y_true = np.atleast_1d(y_true).astype(int)
        num_classes_actual = len(self.class_weights)
        y_pred_reshaped = y_pred.reshape(len(y_true), num_classes_actual)
        prob = softmax(y_pred_reshaped, axis=1)
        y_true_one_hot = np.eye(num_classes_actual)[y_true]
        
        grad = prob - y_true_one_hot
        hess = prob * (1.0 - prob)
        
        sample_weights_internal = self.class_weights[y_true]
        grad = grad * sample_weights_internal[:, np.newaxis]
        hess = hess * sample_weights_internal[:, np.newaxis]
        
        if sample_weight is not None:
            grad = grad * sample_weight[:, np.newaxis]
            hess = hess * sample_weight[:, np.newaxis]
        
        return grad.flatten(), hess.flatten()

# --- [신규 추가] Rho MultiClass Power Scaled Adaptive CE ---
class RhoMultiClassPowerScaledAdaptiveCE:
    """
    다중 클래스용 Power Scaled Adaptive CE + Rho 파라미터
    최다 빈도 클래스를 제외한 나머지 모든 클래스에 rho를 곱함
    """
    def __init__(self, n_classes: int, alpha: float = 1.0, rho: float = 1.0, model_type: str = 'xgboost'):
        self.n_classes = n_classes
        self.alpha = alpha
        self.rho = rho
        self.model_type = model_type
        self.class_weights = None

    def initialize(self, y_true: np.ndarray):
        class_counts = _get_class_counts(y_true, self.n_classes)
        num_classes_from_data = len(class_counts)
        
        # 기본 가중치 계산
        raw_weights = 1.0 / np.power(np.log(1 + class_counts + 1e-9), self.alpha)
        
        # 다수 클래스 찾기 및 나머지 클래스에 rho 적용
        majority_class_idx = np.argmax(class_counts)
        for i in range(len(raw_weights)):
             if i != majority_class_idx:
                 raw_weights[i] *= self.rho

        # 가중치 정규화
        self.class_weights = raw_weights / np.sum(raw_weights) * num_classes_from_data

    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> tuple:
        y_true = np.atleast_1d(y_true).astype(int); num_classes_actual = len(self.class_weights)
        y_pred_reshaped = y_pred.reshape(len(y_true), num_classes_actual); prob = softmax(y_pred_reshaped, axis=1)
        y_true_one_hot = np.eye(num_classes_actual)[y_true]
        grad = prob - y_true_one_hot
        if self.model_type == 'lightgbm': hess = prob
        else: hess = prob * (1.0 - prob)
        sample_weights_internal = self.class_weights[y_true]
        grad = grad * sample_weights_internal[:, np.newaxis]; hess = hess * sample_weights_internal[:, np.newaxis]
        if sample_weight is not None: grad = grad * sample_weight[:, np.newaxis]; hess = hess * sample_weight[:, np.newaxis]
        return grad.flatten(), hess.flatten()