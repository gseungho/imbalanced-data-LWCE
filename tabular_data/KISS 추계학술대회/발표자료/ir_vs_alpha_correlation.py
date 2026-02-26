import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# --- 1. 데이터셋별 IR(불균형 비율) 정의 ---
# (이진 분류만 우선 사용, Max/Min 비율 기준)
dataset_ir_map = {
    'credit_card_fraud': 587,  # (0.17%)
    'aps_failure': 59,       # (1.7%)
    'secom': 14,             # (~6.8%)
    'page_blocks': 175,      # (Multi, Max/Min)
    'bank_marketing': 7.5,   # (~11.7%)
    'yeast': 93,             # (Multi, Max/Min)
    'steel_faults': 12,      # (Multi, Max/Min)
    'credit_card_default': 3.5,  # (~22%)
    'german_credit': 2.3,    # (30%)
    'telco_churn': 2.8,      # (~26.5%)
    'glass': 8.4             # (Multi, Max/Min)
}

# --- 2. 최적화된 Alpha 값 로드 ---
json_path = 'optimized_parameters_xgb_linear.json' # 100-trial 결과 JSON
try:
    with open(json_path, 'r') as f:
        optimized_params = json.load(f)
except Exception as e:
    print(f"JSON 파일 로드 실패: {e}")
    # (실패 시 더미 데이터 사용)
    optimized_params = {
        'credit_card_fraud_power_scaled_adaptive_ce': {'params': {'loss_power_alpha': 2.9}},
        'aps_failure_power_scaled_adaptive_ce': {'params': {'loss_power_alpha': 1.8}},
        'bank_marketing_power_scaled_adaptive_ce': {'params': {'loss_power_alpha': 1.2}},
        'telco_churn_power_scaled_adaptive_ce': {'params': {'loss_power_alpha': 1.1}},
    }

# --- 3. 데이터프레임 생성 ---
results = []
for key, value in optimized_params.items():
    if 'power_scaled_adaptive_ce' in key: # PLWCE 결과만 추출
        dataset_name = key.replace('_power_scaled_adaptive_ce', '')
        if dataset_name in dataset_ir_map:
            alpha = value['params'].get('loss_power_alpha')
            if alpha is not None:
                results.append({
                    'Dataset': dataset_name,
                    'IR (Log Scale)': np.log10(dataset_ir_map[dataset_name]), # X축은 로그 스케일로
                    'Optimal Alpha ($\mathbf{\\alpha}$)' : alpha
                })

df_plot = pd.DataFrame(results)

if df_plot.empty:
    print("분석할 PLWCE 데이터를 찾지 못했습니다. JSON 파일 내용을 확인해주세요.")
else:
    # --- 4. 시각화 (Scatter Plot with Regression Line) ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 산점도와 함께 회귀선(regplot)을 그려 경향성(상관관계)을 보여줌
    sns.regplot(data=df_plot, 
                x='IR (Log Scale)', 
                y='Optimal Alpha ($\mathbf{\\alpha}$)',
                scatter_kws={'s': 100, 'alpha': 0.8}, # 점 크기
                line_kws={'color': 'red', 'linestyle': '--'}) # 회귀선 스타일

    # 각 점에 데이터셋 이름 표시
    for i, row in df_plot.iterrows():
        plt.text(row['IR (Log Scale)'] + 0.05, row['Optimal Alpha ($\mathbf{\\alpha}$)'], 
                 row['Dataset'].replace('_', ' ').title(), fontsize=9)
    
    # --- 5. 꾸미기 ---
    plt.title('Optimal $\\alpha$ vs. Imbalance Ratio (IR)', fontsize=16, fontweight='bold')
    plt.xlabel('Imbalance Ratio (IR) - Log10 Scale (Higher = More Imbalanced)', fontsize=12)
    plt.ylabel('Optimal $\\alpha$ for PLWCE', fontsize=12)
    
    # X축 눈금을 로그 스케일 값으로 보기 좋게 변경 (예: 10^1, 10^2)
    xtick_values = np.arange(0, 4)
    plt.xticks(xtick_values, [f'10^{x}' for x in xtick_values])
    
    plt.tight_layout()
    plt.savefig('ir_vs_alpha_correlation.pdf', bbox_inches='tight')
    print("IR vs Alpha 상관관계 그래프가 저장되었습니다.")
    plt.show()