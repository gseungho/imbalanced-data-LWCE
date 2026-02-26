import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 설정 ---
N_maj = 10000  # 다수 클래스 샘플 수 (기준점)
n_range = np.arange(1, N_maj + 1) # 1부터 10000까지

# --- 가중치 함수 정의 ---
def get_wce_weights(n):
    raw_weights = 1.0 / n
    # 다수 클래스(N_maj)의 가중치를 1.0으로 정규화
    return raw_weights / raw_weights[-1]

def get_wce_sq_weights(n):
    raw_weights = 1.0 / np.sqrt(n)
    # 다수 클래스(N_maj)의 가중치를 1.0으로 정규화
    return raw_weights / raw_weights[-1]

def get_lwce_weights(n, c=1):
    raw_weights = 1.0 / np.log(n + c)
    # 다수 클래스(N_maj)의 가중치를 1.0으로 정규화
    return raw_weights / raw_weights[-1]

def get_plwce_weights_norm(n, alpha, c=1):
    raw_weights = 1.0 / np.power(np.log(n + c), alpha)
    # 다수 클래스(N_maj)의 가중치를 1.0으로 정규화 (상대적 비교를 위해)
    return raw_weights / raw_weights[-1]

# --- 데이터 계산 ---
w_wce_norm = get_wce_weights(n_range)
w_wce_sq_norm = get_wce_sq_weights(n_range)
w_lwce_norm = get_lwce_weights(n_range)
w_plwce_alpha_2_0 = get_plwce_weights_norm(n_range, alpha=2.0) # 기본 LWCE
w_plwce_alpha_3_0 = get_plwce_weights_norm(n_range, alpha=3.0) # Sharp
w_plwce_alpha_4_0 = get_plwce_weights_norm(n_range, alpha=4.0) # Soft

# --- 시각화 ---
sns.set_style("whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 6))

# 메인 그래프 (선형 스케일로 폭발적인 차이 강조)
zoom_idx = 500
ax1.plot(n_range[:zoom_idx], w_wce_norm[:zoom_idx], 
         color='#d62728', linewidth=3, label=r'Standard WCE ($1/n_i$)') # raw string 사용
ax1.plot(n_range[:zoom_idx], w_wce_sq_norm[:zoom_idx], 
         color='#ff7f0e', linewidth=3, label=r'Sqrt WCE ($1/\sqrt{n_i}$)') # \sqrt{...} 로 수정
ax1.plot(n_range[:zoom_idx], w_lwce_norm[:zoom_idx], 
         color='#1f77b4', linewidth=3, label=r'Proposed LWCE ($1/\log(n_i+1)$)')
ax1.plot(n_range[:zoom_idx], w_plwce_alpha_2_0[:zoom_idx], 
         color='#9467bd', linewidth=2, linestyle='--', label=r'PLWCE ($\alpha=2.0$)')
ax1.plot(n_range[:zoom_idx], w_plwce_alpha_3_0[:zoom_idx], 
         color='#bcbd22', linewidth=2, linestyle='--', label=r'PLWCE ($\alpha=3.0$)')
ax1.plot(n_range[:zoom_idx], w_plwce_alpha_4_0[:zoom_idx], 
         color='#8c564b', linewidth=2, linestyle='--', label=r'PLWCE ($\alpha=4.0$)')

# 기준선 (Majority Class Weight = 1.0)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Majority Class Baseline (1.0)')

# 꾸미기
ax1.set_title('Relative Weight Explosion: Minority vs. Majority', fontsize=16, fontweight='bold')
ax1.set_xlabel('Number of Samples ($n_i$)', fontsize=14)
ax1.set_ylabel('Relative Weight (Normalized to $N_{maj}=10k$)', fontsize=14)
ax1.legend(fontsize=13)
ax1.set_xlim(0, 500)  # 0~500 구간 확대
ax1.set_ylim(0, 100) # y축을 제한하여 WCE가 하늘을 뚫고 올라가는 것을 보여줌 (실제 값은 10000)

# WCE 폭발 지점에 텍스트 추가
ax1.text(150, 80, 'WCE Explodes\n(Reach 10,000 at $n=1$)', color='#d62728', fontsize=12, fontweight='bold')
ax1.text(50, 40, 'Sqrt WCE Explodes\n(Reach ~100 at $n=1$)', color='#ff7f0e', fontsize=12, fontweight='bold')
ax1.text(100, 15, 'LWCE remains stable\n(Max ~13.3)', color='#1f77b4', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('wce_vs_lwce_relative_v3.pdf', bbox_inches='tight')
print("상대적 가중치 비교 그래프가 저장되었습니다.")