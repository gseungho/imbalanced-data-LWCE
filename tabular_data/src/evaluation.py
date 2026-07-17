# src/evaluation.py

import os # 파일 경로 관리를 위한 모듈
import numpy as np # 수치 연산을 위한 모듈
import pandas as pd # 데이터 조작 및 분석을 위한 모듈
import seaborn as sns # 고급 시각화를 위한 모듈
import matplotlib.pyplot as plt # 그래프 생성을 위한 모듈
from sklearn.metrics import ( # scikit-learn 라이브러리에서 다양한 평가 지표 함수 임포트
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)
from sklearn.preprocessing import label_binarize # [추가] 다중 클래스 PR AUC 계산을 위해 임포트

# 성능 평가를 수행하는 메인 함수
def evaluate_classification_model(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  n_classes: int,  # 전체 클래스 개수 (테스트셋에 없는 클래스 처리를 위해)
                                  model_name: str, loss_name: str, dataset_name: str,
                                  save_path: str = None):
    
    is_multiclass = n_classes > 2 # 클래스 개수가 2를 초과하면 다중 클래스로 판단
    
    all_labels = list(range(n_classes)) # 전체 클래스 레이블 리스트 생성

    if is_multiclass: # 다중 클래스일 경우
        y_pred = np.argmax(y_pred_proba, axis=1) # 가장 확률 높은 클래스의 인덱스를 예측값으로 선택
    else: # 이진 분류일 경우
        y_pred = (y_pred_proba > 0.5).astype(int) # 확률이 0.5 이상이면 1(Positive), 아니면 0(Negative)으로 예측

    accuracy = accuracy_score(y_true, y_pred) # 정확도 계산
    
    if is_multiclass: # 다중 클래스 지표 계산
        # average='macro': 각 클래스별 지표를 계산한 후 단순 평균 (불균형 데이터에 적합)
        # zero_division=0: 특정 클래스 예측이 없어 분모가 0이 될 경우 경고 없이 0으로 처리
        # labels=all_labels: y_test에 없는 클래스도 포함하여 전체 클래스 기준으로 지표 계산
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=all_labels)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=all_labels)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=all_labels)
        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro', labels=all_labels)
        
        # --- [수정] 다중 클래스 PR AUC (Macro) 계산 ---
        # y_true를 [0, 1, 2] 같은 1D 배열에서 (n_samples, n_classes) 형태의 원-핫 인코딩으로 변환
        y_true_binarized = label_binarize(y_true, classes=all_labels)
        # OvR 방식으로 각 클래스의 PR AUC를 계산하고 'macro' 평균을 냅니다.
        pr_auc = average_precision_score(y_true_binarized, y_pred_proba, average='macro')
        # ---------------------------------------------
        
    else: # 이진 분류 지표 계산
        precision = precision_score(y_true, y_pred, zero_division=0) # 정밀도
        recall = recall_score(y_true, y_pred, zero_division=0) # 재현율
        f1 = f1_score(y_true, y_pred, zero_division=0) # F1 점수
        roc_auc = roc_auc_score(y_true, y_pred_proba) # ROC AUC 점수
        pr_auc = average_precision_score(y_true, y_pred_proba) # PR AUC 점수

    print("\n" + "="*50)
    print(f"평가 결과: [데이터셋: {dataset_name}, 모델: {model_name}, 손실함수: {loss_name}]")
    print("="*50)
    print(f"정확도 (Accuracy)        : {accuracy:.4f}")
    print(f"정밀도 (Precision)       : {precision:.4f}")
    print(f"재현율 (Recall)          : {recall:.4f}")
    print(f"F1 점수 (F1-Score)       : {f1:.4f}")
    print(f"ROC AUC                  : {roc_auc:.4f}")
    
    # [수정] 이제 pr_auc는 항상 숫자형이므로 조건문 변경
    if pd.notna(pr_auc):
        print(f"평균 정밀도 (PR AUC)     : {pr_auc:.4f}") # PR AUC 출력
    else:
        print(f"평균 정밀도 (PR AUC)     : N/A (계산 실패)")
        
    print("="*50 + "\n") # 구분선 출력

    cm = confusion_matrix(y_true, y_pred, labels=all_labels) # 혼동 행렬을 전체 클래스 기준으로 생성
    plt.figure(figsize=(8, 6)) # 그래프 크기 설정
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels) # 혼동 행렬 시각화
    plt.xlabel('Predicted Label'); plt.ylabel('True Label') # x, y축 라벨 설정
    plt.title(f'Confusion Matrix\nDataset: {dataset_name}\nModel: {model_name}, Loss: {loss_name}') # 그래프 제목 설정
    
    if save_path: # 저장 경로가 지정되었다면
        os.makedirs(save_path, exist_ok=True) # 경로가 없으면 생성
        filename = f"{dataset_name}_{model_name}_{loss_name}_confusion_matrix.png" # 파일명 생성
        full_path = os.path.join(save_path, filename) # 전체 경로 조합
        plt.savefig(full_path, dpi=300, bbox_inches='tight') # 그래프를 이미지 파일로 저장
        print(f"혼동 행렬 그래프가 '{full_path}'에 저장되었습니다.")
    plt.show() # 화면에 그래프 표시

    metrics = {'Dataset': dataset_name, 'Model': model_name, 'Loss': loss_name, 'Accuracy': accuracy,
               'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'ROC_AUC': roc_auc,
               'PR_AUC': pr_auc} # [수정] -1 대신 실제 pr_auc 값 저장
    return metrics # 최종 지표 딕셔너리 반환