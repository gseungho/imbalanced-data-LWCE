# src/optuna_tuner_xgb_linear.py
# XGBoost의 'gblinear' 부스터를 사용한 선형 모델 최적화
# 커스텀 손실 함수 완벽 지원

import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.special import expit, softmax
import warnings
warnings.filterwarnings('ignore')

try:
    import custom_losses
except ImportError:
    print("⚠️ custom_losses.py를 임포트할 수 없습니다.")

class OptunaHyperparameterTuner:
    """
    XGBoost gblinear (선형 모델) + 커스텀 손실 함수 최적화 클래스
    """
    
    def __init__(self, n_trials=50, cv_folds=5, random_state=42, 
                 direction='maximize', metric='f1_macro'):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.direction = direction
        self.metric = metric
        self.study = None
        self.best_params = None
        self.best_score = None
        
    def _get_xgboost_param_space(self, trial, is_multiclass, n_classes):
        """
        XGBoost gblinear 하이퍼파라미터 탐색 공간
        """
        params = {
            'booster': 'gblinear',  # 선형 모델
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  # L1 정규화
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),  # L2 정규화
            'random_state': self.random_state,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        if is_multiclass:
            params['num_class'] = n_classes
            params['eval_metric'] = 'mlogloss'
            
        return params
    
    def _get_loss_param_space(self, trial, loss_name):
        """
        커스텀 손실 함수 하이퍼파라미터 탐색 공간
        """
        loss_params = {}
        
        if 'focal' in loss_name.lower():
            # Focal Loss는 민감하므로 처음 몇 번은 검증된 조합 시도
            if trial.number == 0:
                loss_params['alpha'] = 0.25  # 논문 기본값
                loss_params['gamma'] = 2.0   # 논문 기본값
            elif trial.number == 1:
                loss_params['alpha'] = 0.5
                loss_params['gamma'] = 2.0
            elif trial.number == 2:
                loss_params['alpha'] = 0.25
                loss_params['gamma'] = 1.5
            else:
                loss_params['alpha'] = trial.suggest_float('loss_alpha', 0.2, 0.8)
                loss_params['gamma'] = trial.suggest_float('loss_gamma', 0.5, 2.0)
            
        if 'class_balanced' in loss_name.lower() or 'cb' in loss_name.lower():
            loss_params['beta'] = trial.suggest_float('loss_beta', 0.9, 0.999)
            
        if loss_name == 'power_scaled_adaptive_ce':
            # Adaptive CE의 power scaling - 처음 2번은 고정값 시도
            if trial.number == 0:
                loss_params['power_alpha'] = 1.0  # adaptive_ce와 동일
            elif trial.number == 1:
                loss_params['power_alpha'] = 1.5  # 좀 더 강한 가중치
            else:
                loss_params['power_alpha'] = trial.suggest_float('loss_power_alpha', 1.5, 3.0)
        
        if loss_name == 'power_scaled_inv_ce':
            # 처음 3번은 고정값(0.5, 1.0, 0.7)을 강제로 시도
            if trial.number == 0:
                loss_params['inv_power'] = 0.5  # sqrt와 동일
            elif trial.number == 1:
                loss_params['inv_power'] = 1.0  # inverse와 동일
            elif trial.number == 2:
                loss_params['inv_power'] = 0.7  # 중간값
            else:
                # 이후부터는 자유롭게 탐색
                loss_params['inv_power'] = trial.suggest_float('loss_inv_power', 0.3, 0.9)
        
        # [신규] Rho 버전 탐색 공간
        if 'rho_power_scaled' in loss_name.lower():
             # alpha 탐색 (기존과 동일하게 설정하거나 필요시 조정)
             if 'adaptive' in loss_name.lower():
                 loss_params['power_alpha'] = trial.suggest_float('loss_power_alpha', 1.0, 3.0)
             else: # inverse
                 loss_params['inv_power'] = trial.suggest_float('loss_inv_power', 0.3, 1.5)
             
             # rho 탐색 추가 (1.0 ~ 5.0 범위 예시)
             loss_params['rho'] = trial.suggest_float('loss_rho', 1.0, 5.0)
            
        return loss_params
    
    def _create_loss_function(self, loss_name, loss_params, is_multiclass, n_classes):
        """
        손실 함수 객체 생성
        """
        if loss_name == 'standard_ce':
            return None
        
        # 이진 분류 손실 함수
        if not is_multiclass:
            if loss_name == 'weighted_ce_inv':
                return custom_losses.WeightedCrossEntropy('inverse_frequency')
            
            elif loss_name == 'weighted_ce_inv_sqrt':
                return custom_losses.WeightedCrossEntropy('inverse_sqrt_frequency')
            
            elif loss_name == 'class_balanced_ce':
                # ClassBalancedLoss with base sigmoid CE
                base_sigmoid_ce = lambda yt, yp: (expit(yp) - yt, expit(yp) * (1 - expit(yp)))
                return custom_losses.ClassBalancedLoss(
                    base_sigmoid_ce, 
                    loss_params.get('beta', 0.999)
                )
            
            elif loss_name == 'focal_loss':
                return custom_losses.FocalLoss(
                    alpha=loss_params.get('alpha', 0.25),
                    gamma=loss_params.get('gamma', 2.0)
                )
            
            elif loss_name == 'cb_focal_loss':
                return custom_losses.ClassBalancedFocalLoss(
                    beta=loss_params.get('beta', 0.999),
                    gamma=loss_params.get('gamma', 2.0)
                )
            
            elif loss_name == 'adaptive_ce':
                return custom_losses.AdaptiveWeightedCrossEntropy()
            
            elif loss_name == 'power_scaled_adaptive_ce':
                return custom_losses.PowerScaledAdaptiveCE(
                    alpha=loss_params.get('power_alpha', 1.0)
                )
            
            elif loss_name == 'power_scaled_inv_ce':
                return custom_losses.PowerScaledInverseCE(
                    alpha=loss_params.get('inv_power', 0.5)
                )
            elif loss_name == 'rho_power_scaled_adaptive_ce':
                return custom_losses.RhoPowerScaledAdaptiveCE(
                    alpha=loss_params.get('power_alpha', 1.0),
                    rho=loss_params.get('rho', 1.0)
                )
            elif loss_name == 'rho_power_scaled_inv_ce':
                 return custom_losses.RhoPowerScaledInverseCE(
                    alpha=loss_params.get('inv_power', 0.5),
                    rho=loss_params.get('rho', 1.0)
                )
        
        # 다중 클래스 손실 함수
        else:
            if loss_name == 'weighted_ce_inv':
                return custom_losses.MultiClassWeightedCrossEntropy(
                    n_classes, 'inverse_frequency', model_type='xgboost'
                )
            
            elif loss_name == 'weighted_ce_inv_sqrt':
                return custom_losses.MultiClassWeightedCrossEntropy(
                    n_classes, 'inverse_sqrt_frequency', model_type='xgboost'
                )
            
            elif loss_name == 'class_balanced_ce':
                # MultiClassClassBalancedLoss with base softmax CE
                base_softmax_ce = lambda yt, yp: (
                    (softmax(yp.reshape(-1, n_classes), axis=1) - 
                     np.eye(n_classes)[yt.astype(int)]).flatten(),
                    (softmax(yp.reshape(-1, n_classes), axis=1) * 
                     (1 - softmax(yp.reshape(-1, n_classes), axis=1))).flatten()
                )
                return custom_losses.MultiClassClassBalancedLoss(
                    n_classes, 
                    base_softmax_ce, 
                    loss_params.get('beta', 0.999)
                )
            
            elif loss_name == 'focal_loss':
                return custom_losses.MultiClassFocalLoss(
                    n_classes,
                    alpha=loss_params.get('alpha', 0.25),
                    gamma=loss_params.get('gamma', 2.0),
                    model_type='xgboost'
                )
            
            elif loss_name == 'cb_focal_loss':
                return custom_losses.MultiClassClassBalancedFocalLoss(
                    n_classes,
                    beta=loss_params.get('beta', 0.999),
                    gamma=loss_params.get('gamma', 2.0),
                    model_type='xgboost'
                )
            
            elif loss_name == 'adaptive_ce':
                return custom_losses.MultiClassAdaptiveCrossEntropy(
                    n_classes, model_type='xgboost'
                )
            
            elif loss_name == 'power_scaled_adaptive_ce':
                return custom_losses.MultiClassPowerScaledAdaptiveCE(
                    n_classes,
                    alpha=loss_params.get('power_alpha', 1.0),
                    model_type='xgboost'
                )
            
            elif loss_name == 'power_scaled_inv_ce':
                return custom_losses.MultiClassPowerScaledInverseCE(
                    n_classes,
                    alpha=loss_params.get('inv_power', 0.5)
                )
            # [신규] Rho 버전 추가
            elif loss_name == 'rho_power_scaled_adaptive_ce':
                return custom_losses.RhoMultiClassPowerScaledAdaptiveCE(
                    n_classes,
                    alpha=loss_params.get('power_alpha', 1.0),
                    rho=loss_params.get('rho', 1.0),
                    model_type='xgboost'
                )
            elif loss_name == 'rho_power_scaled_inv_ce':
                return custom_losses.RhoMultiClassPowerScaledInverseCE(
                    n_classes,
                    alpha=loss_params.get('inv_power', 0.5),
                    rho=loss_params.get('rho', 1.0)
                )
        
        return None
    
    def _objective(self, trial, X_train, y_train, loss_name='standard_ce'):
        """
        Optuna objective 함수
        """
        # 라벨을 0부터 연속적으로 재인코딩
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        # 다중 클래스 여부 확인
        unique_classes = np.unique(y_train_encoded)
        is_multiclass = len(unique_classes) > 2
        
        if is_multiclass:
            n_classes = len(unique_classes) 
        else:
            n_classes = len(unique_classes)
        
        # 하이퍼파라미터 샘플링
        xgb_params = self._get_xgboost_param_space(trial, is_multiclass, n_classes)
        loss_params = self._get_loss_param_space(trial, loss_name)
        
        # 손실 함수 생성
        loss_obj = self._create_loss_function(
            loss_name, loss_params, is_multiclass, n_classes
        )
        
        # Objective 설정
        if loss_name == 'standard_ce':
            if is_multiclass:
                xgb_params['objective'] = 'multi:softprob'
            else:
                xgb_params['objective'] = 'binary:logistic'
        else:
            # 커스텀 손실 함수 초기화
            if loss_obj is not None and hasattr(loss_obj, 'initialize'):
                loss_obj.initialize(y_train_encoded)
            xgb_params['objective'] = loss_obj.compute_grad_hess if loss_obj else 'binary:logistic'
        
        # 모델 생성
        model = xgb.XGBClassifier(**xgb_params)
        
        # 교차 검증
        cv = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        try:
            scores = cross_val_score(
                model, X_train, y_train_encoded,
                cv=cv, scoring=self.metric, n_jobs=-1
            )
            score = scores.mean()
            
            if score == 0.0:
                print(f"⚠️ Trial {trial.number}: Score is 0.0! Scores: {scores}")
            
            return score
        except Exception as e:
            print(f"❌ Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0 if self.direction == 'maximize' else float('inf')
    
    def optimize(self, X_train, y_train, loss_name='standard_ce', 
                 study_name=None, show_progress_bar=True):
        """
        하이퍼파라미터 최적화 실행
        """
        if study_name is None:
            study_name = f"xgb_linear_{loss_name}_{self.metric}"
        
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        print(f"\n{'='*60}")
        print(f"Optuna 하이퍼파라미터 최적화 시작")
        print(f"모델: XGBoost gblinear (선형 모델)")
        print(f"손실 함수: {loss_name}")
        print(f"평가 지표: {self.metric}")
        print(f"시행 횟수: {self.n_trials}")
        print(f"{'='*60}\n")
        
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, loss_name),
            n_trials=self.n_trials,
            show_progress_bar=show_progress_bar,
            timeout=3600  # 1시간 제한
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"\n{'='*60}")
        print(f"최적화 완료!")
        print(f"완료된 시행: {len(self.study.trials)}회")
        print(f"최고 점수: {self.best_score:.4f}")
        print(f"최적 파라미터:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"{'='*60}\n")
        
        return self.best_params
    
    def get_best_model(self, X_train, y_train, loss_name='standard_ce'):
        """
        최적 하이퍼파라미터로 학습된 모델 반환
        """
        if self.best_params is None:
            raise ValueError("먼저 optimize() 메서드를 실행해주세요.")
        
        # 라벨을 0부터 연속적으로 재인코딩
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        # 다중 클래스 여부 확인
        unique_classes = np.unique(y_train_encoded)
        is_multiclass = len(unique_classes) > 2
        
        if is_multiclass:
            n_classes = len(unique_classes)  # max+1이 아닌 실제 클래스 개수
        else:
            n_classes = len(unique_classes)
        
        # 손실 함수 관련 파라미터 분리
        loss_params = {k.replace('loss_', ''): v for k, v in self.best_params.items() if k.startswith('loss_')}
        xgb_params = {k: v for k, v in self.best_params.items() if not k.startswith('loss_')}
        
        # 추가 파라미터 설정
        xgb_params['booster'] = 'gblinear'
        xgb_params['random_state'] = self.random_state
        xgb_params['use_label_encoder'] = False
        
        if is_multiclass:
            xgb_params['num_class'] = n_classes
            xgb_params['eval_metric'] = 'mlogloss'
        else:
            xgb_params['eval_metric'] = 'logloss'
        
        # 손실 함수 생성
        loss_obj = self._create_loss_function(
            loss_name, loss_params, is_multiclass, n_classes
        )
        
        # Objective 설정
        if loss_name == 'standard_ce':
            if is_multiclass:
                xgb_params['objective'] = 'multi:softprob'
            else:
                xgb_params['objective'] = 'binary:logistic'
        else:
            if loss_obj is not None and hasattr(loss_obj, 'initialize'):
                loss_obj.initialize(y_train_encoded)
            xgb_params['objective'] = loss_obj.compute_grad_hess if loss_obj else 'binary:logistic'
        
        # 모델 생성 및 학습
        print("✨ 최적 파라미터로 최종 모델 학습 중...")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train_encoded)  # 인코딩된 레이블로 학습
        print("✅ 학습 완료!")
        
        # LabelEncoder 저장 (나중에 역변환용)
        model.label_encoder_ = le
        model.n_classes_actual_ = n_classes
        
        return model
    
    def plot_optimization_history(self):
        """최적화 히스토리 플롯"""
        if self.study is None:
            raise ValueError("먼저 optimize() 메서드를 실행해주세요.")
        return optuna.visualization.plot_optimization_history(self.study)
    
    def plot_param_importances(self):
        """하이퍼파라미터 중요도 플롯"""
        if self.study is None:
            raise ValueError("먼저 optimize() 메서드를 실행해주세요.")
        return optuna.visualization.plot_param_importances(self.study)