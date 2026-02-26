import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. 데이터 준비 ---
# 제공해주신 5개 데이터셋 (불균형 심한 순서)
datasets = ['Credit Card Fraud', 'APS Failure', 'Bank Marketing', 'Credit Card Default', 'German Credit']
# F1-Score 데이터
standard_ce = [0.7287, 0.7537, 0.4544, 0.3568, 0.5714]
adaptive_ce = [0.7473, 0.7712, 0.4887, 0.4876, 0.5909]
plwce =       [0.7986, 0.7831, 0.5474, 0.4913, 0.6146]
# 최적 Alpha 값
alphas =      [2.76,   1.62,   2.96,   2.99,   2.86]

x = np.arange(len(datasets))
width = 0.25  # 막대 너비

# --- 2. 시각화 ---
fig, ax = plt.subplots(figsize=(12, 7))

# 막대 그리기
# Standard CE (회색조로 베이스라인임을 표현)
rects1 = ax.bar(x - width, standard_ce, width, label='Standard CE', color='#999999', alpha=0.7)
# LWCE (파란색 계열로 1차 개선임을 표현)
rects2 = ax.bar(x, adaptive_ce, width, label='LWCE (Standard Adaptive CE)', color='#1f77b4', alpha=0.8)
# PLWCE (붉은색 계열로 최종 제안 모델임을 강조, 테두리 추가)
rects3 = ax.bar(x + width, plwce, width, label='PLWCE (Power Scaled Adpative CE)', color='#d62728', alpha=1.0, edgecolor='black', linewidth=1.2)

# --- 3. Alpha 값 주석 추가 (핵심!) ---
# PLWCE 막대 위에 최적 alpha 값을 표시하여 "alpha > 1"의 효과를 강조
for i, rect in enumerate(rects3):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + 0.015,
            f'$\\alpha={alphas[i]}$',
            ha='center', va='bottom', fontweight='bold', color='#d62728', fontsize=11)

# --- 4. 꾸미기 ---
ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12, rotation=15, ha='right')
ax.set_ylim(0, 0.95) # y축 범위 여유있게 설정

# 범례
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

# 그리드 및 스타일
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)


# --- 5. 저장 ---
plt.savefig('plwce_alpha_effect_v2.pdf', bbox_inches='tight')
print("Alpha 효과 검증 그래프가 저장되었습니다.")
plt.show()