# src/optuna_tuner_xgb_linear.py
# [최종 수정본]
# 1. fixed_model_params 인자 지원
# 2. cross_val_score 대신 수동 K-Fold 루프를 사용하여 손실 함수 initialize 버그 해결
# 3. LabelEncoder를 클래스 내부에 통합하여 일관성 유지

import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score # f1_score 임포트
from scipy.special import expit, softmax
import warnings
from functools import partial
from sklearn.preprocessing import LabelEncoder
import importlib

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
        self.label_encoder_ = LabelEncoder() # 라벨 인코더 내장
        self.n_classes_actual_ = None

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
             loss_params['loss_alpha'] = trial.suggest_float('loss_alpha', 0.2, 0.8)
             loss_params['loss_gamma'] = trial.suggest_float('loss_gamma', 0.5, 5.0)
            
        if 'class_balanced' in loss_name.lower() or 'cb' in loss_name.lower():
            loss_params['loss_beta'] = trial.suggest_float('loss_beta', 0.9, 0.9999)
            
        if loss_name == 'power_scaled_adaptive_ce': # PLWCE
            loss_params['loss_power_alpha'] = trial.suggest_float('loss_power_alpha', 1.0, 3.5)
        
        if loss_name == 'power_scaled_inv_ce': # PS-WCE
            loss_params['loss_inv_power'] = trial.suggest_float('loss_inv_power', 0.1, 0.9)
        
        if 'rho_power_scaled' in loss_name.lower():
             if 'adaptive' in loss_name.lower():
                 loss_params['loss_power_alpha'] = trial.suggest_float('loss_power_alpha', 0.1, 4.0)
             else:
                 loss_params['loss_inv_power'] = trial.suggest_float('loss_inv_power', 0.1, 1.5)
             loss_params['loss_rho'] = trial.suggest_float('loss_rho', 1.0, 5.0)
             
        return loss_params
    
    def _create_loss_function(self, loss_name, loss_params, is_multiclass, n_classes):
        """
        손실 함수 객체 생성 (loss_params에서 'loss_' 접두사 제거)
        """
        clean_params = {k.replace('loss_', ''): v for k, v in loss_params.items()}
        
        if loss_name == 'standard_ce':
            return None
        
        if not is_multiclass:
            if loss_name == 'weighted_ce_inv':
                return custom_losses.WeightedCrossEntropy('inverse_frequency')
            elif loss_name == 'weighted_ce_inv_sqrt':
                return custom_losses.WeightedCrossEntropy('inverse_sqrt_frequency')
            elif loss_name == 'class_balanced_ce':
                base_sigmoid_ce = lambda yt, yp: (expit(yp) - yt, expit(yp) * (1 - expit(yp)))
                return custom_losses.ClassBalancedLoss(base_sigmoid_ce, beta=clean_params.get('beta', 0.999))
            elif loss_name == 'focal_loss':
                return custom_losses.FocalLoss(alpha=clean_params.get('alpha', 0.25), gamma=clean_params.get('gamma', 2.0))
            elif loss_name == 'cb_focal_loss':
                return custom_losses.ClassBalancedFocalLoss(beta=clean_params.get('beta', 0.999), gamma=clean_params.get('gamma', 2.0))
            elif loss_name == 'adaptive_ce':
                return custom_losses.AdaptiveWeightedCrossEntropy()
            elif loss_name == 'power_scaled_adaptive_ce':
                return custom_losses.PowerScaledAdaptiveCE(alpha=clean_params.get('power_alpha', 1.0))
            elif loss_name == 'power_scaled_inv_ce':
                return custom_losses.PowerScaledInverseCE(alpha=clean_params.get('inv_power', 0.5))
            elif loss_name == 'rho_power_scaled_adaptive_ce':
                 return custom_losses.RhoPowerScaledAdaptiveCE(alpha=clean_params.get('power_alpha', 1.0), rho=clean_params.get('rho', 1.0))
            elif loss_name == 'rho_power_scaled_inv_ce':
                 return custom_losses.RhoPowerScaledInverseCE(alpha=clean_params.get('inv_power', 0.5), rho=clean_params.get('rho', 1.0))

        else: # 다중 클래스
            if loss_name == 'weighted_ce_inv':
                return custom_losses.MultiClassWeightedCrossEntropy(n_classes, 'inverse_frequency', model_type='xgboost')
            elif loss_name == 'weighted_ce_inv_sqrt':
                return custom_losses.MultiClassWeightedCrossEntropy(n_classes, 'inverse_sqrt_frequency', model_type='xgboost')
            elif loss_name == 'class_balanced_ce':
                base_softmax_ce = lambda yt, yp: (
                    (softmax(yp.reshape(-1, n_classes), axis=1) - np.eye(n_classes)[yt.astype(int)]).flatten(),
                    (softmax(yp.reshape(-1, n_classes), axis=1) * (1 - softmax(yp.reshape(-1, n_classes), axis=1))).flatten()
                )
                return custom_losses.MultiClassClassBalancedLoss(n_classes, base_softmax_ce, beta=clean_params.get('beta', 0.999))
            elif loss_name == 'focal_loss':
                return custom_losses.MultiClassFocalLoss(n_classes, alpha=clean_params.get('alpha', 0.25), gamma=clean_params.get('gamma', 2.0), model_type='xgboost')
            elif loss_name == 'cb_focal_loss':
                return custom_losses.MultiClassClassBalancedFocalLoss(n_classes, beta=clean_params.get('beta', 0.999), gamma=clean_params.get('gamma', 2.0), model_type='xgboost')
            elif loss_name == 'adaptive_ce':
                return custom_losses.MultiClassAdaptiveCrossEntropy(n_classes, model_type='xgboost')
            elif loss_name == 'power_scaled_adaptive_ce':
                return custom_losses.MultiClassPowerScaledAdaptiveCE(n_classes, alpha=clean_params.get('power_alpha', 1.0), model_type='xgboost')
            elif loss_name == 'power_scaled_inv_ce':
                return custom_losses.MultiClassPowerScaledInverseCE(n_classes, alpha=clean_params.get('inv_power', 0.5))
            elif loss_name == 'rho_power_scaled_adaptive_ce':
                 return custom_losses.RhoMultiClassPowerScaledAdaptiveCE(n_classes, alpha=clean_params.get('power_alpha', 1.0), rho=clean_params.get('rho', 1.0), model_type='xgboost')
            elif loss_name == 'rho_power_scaled_inv_ce':
                 return custom_losses.RhoMultiClassPowerScaledInverseCE(n_classes, alpha=clean_params.get('inv_power', 0.5), rho=clean_params.get('rho', 1.0))

        raise ValueError(f"알 수 없는 손실 함수: {loss_name}")
    
    def _objective(self, trial, X_train, y_train, loss_name='standard_ce', fixed_params=None):
        """
        Optuna objective 함수
        [수정됨] cross_val_score 대신 수동 K-Fold 루프를 사용하여
        매 Fold마다 손실 함수를 올바르게 초기화합니다.
        """
        
        unique_classes = np.unique(y_train)
        is_multiclass = len(unique_classes) > 2
        n_classes = len(unique_classes)
        
        # 1. 모델 파라미터 가져오기
        if fixed_params:
            xgb_params = fixed_params.copy()
            # 멀티클래스인 경우 fixed_params에 num_class가 없으면 추가
            if is_multiclass and 'num_class' not in xgb_params:
                xgb_params['num_class'] = n_classes
        else:
            # 고정 파라미터가 없으면 튜닝
            xgb_params = self._get_xgboost_param_space(trial, is_multiclass, n_classes)

        # 2. 손실 함수 파라미터 *만* 튜닝 (fixed_params가 있어도 loss param은 튜닝)
        loss_params = self._get_loss_param_space(trial, loss_name)
        
        # 3. K-Fold 교차 검증 (수동 루프)
        cv = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        scores = []
        try:
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                # --- [핵심 수정] ---
                loss_obj = self._create_loss_function(
                    loss_name, loss_params, is_multiclass, n_classes
                )
                
                current_xgb_params = xgb_params.copy()
                if loss_name == 'standard_ce':
                    if is_multiclass:
                        current_xgb_params['objective'] = 'multi:softprob'
                    else:
                        current_xgb_params['objective'] = 'binary:logistic'
                else:
                    if loss_obj is not None and hasattr(loss_obj, 'initialize'):
                        loss_obj.initialize(y_train_fold) 
                    current_xgb_params['objective'] = loss_obj.compute_grad_hess if loss_obj else 'binary:logistic'
                # --- [수정 끝] ---

                model = xgb.XGBClassifier(**current_xgb_params, use_label_encoder=False)
                model.fit(X_train_fold, y_train_fold)
                
                y_pred_proba = model.predict_proba(X_val_fold)
                if is_multiclass:
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
                
                if self.metric == 'f1_macro':
                    scores.append(f1_score(y_val_fold, y_pred, average='macro', zero_division=0))
                else:
                    scores.append(f1_score(y_val_fold, y_pred, average='macro', zero_division=0))

            score = np.mean(scores)
            
            if score == 0.0:
                print(f"⚠️ Trial {trial.number}: Score is 0.0! Scores: {scores}")
            
            return score
        
        except Exception as e:
            print(f"❌ Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0 if self.direction == 'maximize' else float('inf')
    
    # [수정] fixed_model_params 인자 추가
    def optimize(self, X_train, y_train, loss_name='standard_ce', 
                 fixed_model_params=None, study_name=None, show_progress_bar=True):
        """
        하이퍼파라미터 최적화 실행
        """
        
        # 라벨 인코딩 (y가 0부터 시작하도록 보장)
        y_train_encoded = self.label_encoder_.fit_transform(y_train)
        self.n_classes_actual_ = len(np.unique(y_train_encoded))
        
        if study_name is None:
            study_name = f"xgb_linear_{loss_name}_{self.metric}"
            if fixed_model_params:
                study_name += "_fixed" # 연구 이름 구분
        
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
        )
        
        print(f"\n{'='*60}")
        print(f"Optuna 하이퍼파라미터 최적화 시작")
        print(f"모델: XGBoost gblinear (선형 모델)")
        print(f"손실 함수: {loss_name}")
        print(f"평가 지표: {self.metric}")
        print(f"시행 횟수: {self.n_trials}")
        if fixed_model_params:
            print(f"고정 파라미터 사용: {fixed_model_params}")
        print(f"{'='*60}\n")
        
        # [수정] partial로 fixed_model_params 전달
        obj_func = partial(self._objective, X_train=X_train, y_train=y_train_encoded, 
                           loss_name=loss_name, fixed_params=fixed_model_params)
        
        self.study.optimize(
            obj_func,
            n_trials=self.n_trials,
            show_progress_bar=show_progress_bar,
            timeout=3600  # 1시간 제한
        )
        
        # [수정] self.best_params에 결과 저장
        if fixed_model_params:
            self.best_params = fixed_model_params.copy()
            self.best_params.update(self.study.best_params) # 튜닝된 loss 파라미터 추가
        else:
            self.best_params = self.study.best_params # 기존 방식
            
        self.best_score = self.study.best_value
        
        print(f"\n{'='*60}")
        print(f"최적화 완료!")
        print(f"완료된 시행: {len(self.study.trials)}회")
        print(f"최고 점수: {self.best_score:.4f}")
        print(f"최적 파라미터 (손실 함수):")
        for param, value in self.study.best_params.items(): # 모델 파람 제외
            print(f"  {param}: {value}")
        print(f"{'='*60}\n")
        
        return self.best_params
    
    def get_best_model(self, X_train, y_train, loss_name='standard_ce'):
        """
        최적 하이퍼파라미터로 학습된 모델 반환
        """
        if self.best_params is None:
            raise ValueError("먼저 optimize() 메서드를 실행해주세요.")
        
        # 라벨 인코딩 (optimize와 동일하게)
        y_train_encoded = self.label_encoder_.transform(y_train)
        n_classes = self.n_classes_actual_
        is_multiclass = n_classes > 2
        
        # 손실 함수 관련 파라미터 분리
        loss_params = {k: v for k, v in self.best_params.items() if k.startswith('loss_')}
        xgb_params = {k: v for k, v in self.best_params.items() if not k.startswith('loss_')}
        
        # 추가 파라미터 설정 (고정 파라미터가 이미 포함되어 있음)
        xgb_params.setdefault('booster', 'gblinear')
        xgb_params.setdefault('random_state', self.random_state)
        xgb_params.setdefault('use_label_encoder', False)
        
        if is_multiclass:
            xgb_params.setdefault('num_class', n_classes)
            xgb_params.setdefault('eval_metric', 'mlogloss')
        else:
            xgb_params.setdefault('eval_metric', 'logloss')
        
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
        model.label_encoder_ = self.label_encoder_
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