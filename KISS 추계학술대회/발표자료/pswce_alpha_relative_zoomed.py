import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl # matplotlib 모듈 임포트 추가

# --- 설정 ---
N_maj = 10000  # 다수 클래스 샘플 수 (기준점)
n_range = np.arange(1, N_maj + 1) # 1부터 10000까지

# --- PS-WCE 가중치 함수 ( 1 / (n^alpha) ) ---
def get_pswce_weights_norm(n, alpha):
    # n이 정수형일 때 power 연산을 위해 float로 변경
    raw_weights = 1.0 / np.power(n.astype(float), alpha)
    # 다수 클래스(N_maj)의 가중치를 1.0으로 정규화
    norm_factor = raw_weights[-1]
    return raw_weights / norm_factor

# --- 데이터 계산 ---
w_alpha_1_0 = get_pswce_weights_norm(n_range, alpha=1.0) # Standard WCE (n=1에서 10000)
w_alpha_0_5 = get_pswce_weights_norm(n_range, alpha=0.5) # Sqrt WCE (n=1에서 100)
w_alpha_0_25 = get_pswce_weights_norm(n_range, alpha=0.25) # (n=1에서 10)

# --- 시각화 ---
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# x축 범위를 0~200으로 좁혀서 차이를 강조
zoom_idx = 200

# Y축 스케일이 너무 커서 WCE(alpha=1.0)는 보이지 않습니다.
# WCE(sq)와 alpha=0.25를 중심으로 그립니다.
ax.plot(n_range[:zoom_idx], w_alpha_1_0[:zoom_idx],
         color='#d62728', linewidth=2, linestyle=':', label=r'$\alpha=1.0$ (Std. WCE) $\to$ Explodes (Max 10k)')
ax.plot(n_range[:zoom_idx], w_alpha_0_5[:zoom_idx],
         color='#ff7f0e', linewidth=3, linestyle='-', label=r'$\alpha=0.5$ (Sqrt WCE)')
ax.plot(n_range[:zoom_idx], w_alpha_0_25[:zoom_idx],
         color='#2ca02c', linewidth=3, linestyle='--', label=r'$\alpha=0.25$ (Conservative)')

# 기준선
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Majority Baseline (1.0)')

# 꾸미기
ax.set_title(r'Relative Weight of PS-WCE ($1/n_i^\alpha$)', fontsize=16, fontweight='bold')
ax.set_xlabel('Number of Samples ($n_i$)', fontsize=14)
ax.set_ylabel('Relative Weight (Normalized to $N_{maj}$)', fontsize=14)
ax.legend(fontsize=13)

# WCE(sq)가 보이도록 Y축을 110까지 제한
ax.set_xlim(0, 200)
ax.set_ylim(0, 110) 

# 설명 텍스트
ax.text(10, 95, r'$\alpha=0.5 \to$ Start ~100x', color='#ff7f0e', fontsize=12, fontweight='bold')
ax.text(50, 20, r'$\alpha=0.25 \to$ Start ~10x', color='#2ca02c', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('pswce_alpha_relative_zoomed.pdf', bbox_inches='tight')
print("PS-WCE Alpha 비교 그래프가 저장되었습니다.")
plt.show()