# -*- coding: utf-8 -*-
r"""
Proposition 3 (Dynamic-range compression)의 시각화.

y축 = 소수/다수 클래스 raw-weight 비율 w(n_c) / w(n_max), n_max 고정.
이 비율은 mean-normalization(Eq. normalization)에 대해 불변이므로 raw weight로 계산.
공식은 custom_losses.calculate_weights와 정확히 일치 (log1p = log(1+n)).

출력: weight_compression.pdf  (LaTeX \includegraphics 용 vector)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth'] = 0.6
rcParams['pdf.fonttype'] = 42          # TrueType 임베드 (Elsevier 요구)

N_MAX = 10_000                          # 다수 클래스 표본 수 (기준)
N_HI  = 300                             # x축 상한 (n>200이면 곡선 차이 미미 → 흥미 구간만)
n = np.arange(1, N_HI + 1)              # 소수 클래스 표본 수 n_c

def ratio(wfun):
    """w(n_c) / w(n_max) — Proposition 3의 rare-to-frequent 비율."""
    return wfun(n) / wfun(N_MAX)

# custom_losses.calculate_weights와 동일한 raw weight 정의
w_wce   = lambda x: 1.0 / x                          # WCE:    1/n
w_sqce  = lambda x: 1.0 / np.sqrt(x)                 # √-CE:   1/√n
w_lwce  = lambda x: 1.0 / np.log1p(x)               # LWCE:   1/log(1+n)
w_plwce = lambda x, a: 1.0 / np.log1p(x) ** a       # PLWCE:  1/log(1+n)^α
w_eslw  = lambda x, e: 1.0 / (np.log1p(x) + e)      # ES-LWCE:1/(log(1+n)+ε)

fig, ax = plt.subplots(figsize=(6.6, 3.9))

# --- baselines (frequency-based, 비-proposed) ---
ax.plot(n, ratio(w_wce),  color='#c0392b', lw=2.0, ls='-',
        label=r'WCE  $1/n_c$')
ax.plot(n, ratio(w_sqce), color='#e67e22', lw=1.6, ls='-',
        label=r'Inverse-sqrt  $1/\sqrt{n_c}$')

# --- proposed family ---
ax.plot(n, ratio(lambda x: w_plwce(x, 3.0)), color='#8e44ad', lw=1.4, ls=(0, (5, 2)),
        label=r'PLWCE  $\alpha=3$')
ax.plot(n, ratio(lambda x: w_plwce(x, 2.0)), color='#2980b9', lw=1.4, ls=(0, (3, 1.5)),
        label=r'PLWCE  $\alpha=2$')
ax.plot(n, ratio(w_lwce), color='#1a5276', lw=2.4, ls='-',
        label=r'LWCE  $1/\log(1+n_c)$')
ax.plot(n, ratio(lambda x: w_eslw(x, 1.0)), color='#16a085', lw=1.6, ls=(0, (1, 1)),
        label=r'ES-LWCE  $\varepsilon=1$')

ax.axhline(1.0, color='0.4', lw=0.8, ls=(0, (4, 3)))
ax.text(N_HI - 3, 1.05, 'majority baseline', ha='right', va='bottom', fontsize=7.5, color='0.35')

ax.set_yscale('log')
ax.set_xlim(0, N_HI)
ax.set_ylim(0.8, 2e4)
ax.set_xlabel(r'Minority-class sample count $n_c$  (majority fixed at $n_{\max}=10^4$)',
              fontsize=9.5)
ax.set_ylabel(r'Weight ratio $w_{\min}/w_{\max}$', fontsize=9.5)
ax.tick_params(labelsize=8.5)
ax.grid(True, which='major', color='0.9', lw=0.5)
ax.grid(True, which='minor', color='0.96', lw=0.4)
ax.set_axisbelow(True)

# 핵심 수치 주석 (n_c = 1 에서의 비율 = imbalance ρ = 10^4)
ax.annotate(r'WCE $\to \rho=10^4$', xy=(1, w_wce(1)/w_wce(N_MAX)), xytext=(70, 6000),
            fontsize=8, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.7))
ax.annotate(r'LWCE $\approx 13$ (max)', xy=(1, w_lwce(1)/w_lwce(N_MAX)), xytext=(120, 30),
            fontsize=8, color='#1a5276',
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=0.7))

ax.legend(fontsize=7.6, loc='upper right', framealpha=0.95, ncol=1, handlelength=2.4)
fig.tight_layout(pad=0.4)
fig.savefig('weight_compression.pdf', bbox_inches='tight')
print('saved weight_compression.pdf')

# 검증: 논문에 쓸 핵심 수치 출력
print(f"n_c=1 비율 — WCE={w_wce(1)/w_wce(N_MAX):.0f}, "
      f"sqrt={w_sqce(1)/w_sqce(N_MAX):.1f}, "
      f"LWCE={w_lwce(1)/w_lwce(N_MAX):.2f}, "
      f"PLWCE(a=2)={w_plwce(1,2)/w_plwce(N_MAX,2):.1f}, "
      f"ES-LWCE(e=1)={w_eslw(1,1)/w_eslw(N_MAX,1):.2f}")
