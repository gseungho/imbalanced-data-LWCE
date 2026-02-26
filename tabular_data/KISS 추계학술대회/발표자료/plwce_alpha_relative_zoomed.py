import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 설정 ---
N_maj = 10000  # 다수 클래스 샘플 수 (기준점)
n_range = np.arange(1, N_maj + 1) # 1부터 10000까지

# --- PLWCE 가중치 함수 ---
def get_plwce_weights_norm(n, alpha, c=1):
    raw_weights = 1.0 / np.power(np.log(n + c), alpha)
    # 다수 클래스(N_maj)의 가중치를 1.0으로 정규화 (상대적 비교를 위해)
    return raw_weights / raw_weights[-1]

# --- 데이터 계산 ---
w_alpha_1_0 = get_plwce_weights_norm(n_range, alpha=1.0) # 기본 LWCE
w_alpha_0_5 = get_plwce_weights_norm(n_range, alpha=0.5) # Soft
w_alpha_2_0 = get_plwce_weights_norm(n_range, alpha=2.0) # Sharp (소수 클래스 집중)
w_alpha_4_0 = get_plwce_weights_norm(n_range, alpha=4.0)

# --- 시각화 ---
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# x축 범위를 0~200으로 더 좁혀서 차이를 강조
zoom_idx = 100
ax.plot(n_range[:zoom_idx], w_alpha_4_0[:zoom_idx],
         color="#ff1c1c", linewidth=2, linestyle='-', label=r'$\alpha=4.0$ (extremely Focus on Minority)')
ax.plot(n_range[:zoom_idx], w_alpha_2_0[:zoom_idx],
         color="#f7d200", linewidth=2, linestyle='-', label=r'$\alpha=2.0$ (Focus on Minority)')
ax.plot(n_range[:zoom_idx], w_alpha_1_0[:zoom_idx],
         color='#1f77b4', linewidth=2, linestyle='-', label=r'$\alpha=1.0$ (Standard LWCE)')
ax.plot(n_range[:zoom_idx], w_alpha_0_5[:zoom_idx],
         color='#2ca02c', linewidth=2, linestyle='--', label=r'$\alpha=0.5$ (Conservative)')

# 기준선
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Majority Baseline (1.0)')

# 꾸미기
ax.set_title(r'Impact of $\alpha$ on Relative Weight (Zoomed-in: $n_i \in [0, 100]$)', fontsize=16, fontweight='bold')
ax.set_xlabel('Number of Samples ($n_i$)', fontsize=14)
ax.set_ylabel('Relative Weight (Normalized to $N_{maj}$)', fontsize=14)
ax.legend(fontsize=13)

# x축 범위를 100까지로 제한하여 앞부분의 차이를 확대
ax.set_xlim(0, 100)
ax.set_ylim(0, 180) # alpha=2.0의 최대값이 약 177이므로 이 범위는 유지

# 설명 텍스트 위치 조정 (줌인 된 좌표에 맞게)
ax.text(20, 130, r'$\alpha=4.0 \to$ Start ~31Kx', color='#ff1c1c', fontsize=12, fontweight='bold')
ax.text(5, 50, r'$\alpha=2.0 \to$ Start ~177x', color='#f7d200', fontsize=12, fontweight='bold')
ax.text(20, 25, r'$\alpha=1.0 \to$ Start ~13x', color='#1f77b4', fontsize=12, fontweight='bold')
ax.text(50, 12, r'$\alpha=0.5 \to$ Start ~3.6x', color='#2ca02c', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plwce_alpha_relative_zoomed.pdf', bbox_inches='tight')
print("Alpha 상대적 비교 그래프(Zoomed)가 저장되었습니다.")