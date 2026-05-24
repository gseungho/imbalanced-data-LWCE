# Introduction Revision Notes
> **Paper:** Log-Scale Class Weighting for Robust Imbalanced Classification  
> **Based on:** CIFAR-10-LT / CIFAR-100-LT experimental results (5 seeds × 3 IRs × 6 losses)

---

## Summary of Changes

| # | Location | Issue | Action |
|---|----------|-------|--------|
| 1 | CB/Focal critique | Focal outperforms CE on CIFAR-10-LT IR=100 | Separate CB and Focal claims |
| 2 | CB Loss detail | CB beats CE at IR=10/50; "consistently" is inaccurate | Limit claim to IR=100 only |
| 3 | CIFAR-10 result | Focal/PWCE within 0.002 of PLWCE; only CB degrades | Add Focal/PWCE parity, isolate CB |
| 4 | Result summary | "CB and Focal both degrade" contradicts CIFAR-10 Focal result | Split into separate sentences |
| 5 | Contribution 3 | "Better than Focal" overstated for CIFAR-10 | Limit Focal comparison to CIFAR-100 |

---

## Revision 1 — CB Loss / Focal Loss Critique

### Before
```
CB Loss and Focal Loss, introduce additional hyperparameters
and can degrade below standard cross-entropy (CE) performance
at high imbalance ratios.
```

### After
```
CB Loss introduces additional hyperparameters and consistently
degrades below CE at severe imbalance (IR=100). Focal Loss
exhibits dataset-dependent behavior: while competitive on
CIFAR-10-LT, it fails to improve over CE on CIFAR-100-LT
at IR=100.
```

### Evidence
| Dataset | Loss | IR=100 F1 | vs CE |
|---------|------|-----------|-------|
| CIFAR-10-LT | Focal | 0.4565 | **+0.0259** ↑ |
| CIFAR-10-LT | CE | 0.4306 | — |
| CIFAR-100-LT | Focal | 0.1689 | **−0.0097** ↓ |
| CIFAR-100-LT | CE | 0.1786 | — |

Focal outperforms CE on CIFAR-10-LT IR=100, so grouping it with CB Loss as universally degrading is factually incorrect.

---

## Revision 2 — CB Loss Detail Sentence

### Before
```
CB Loss consistently underperforms CE at high imbalance
ratios---degrading to F1-Macro of 0.406 against CE's 0.431
on CIFAR-10-LT at IR = 100, and 0.158 against 0.179 on
CIFAR-100-LT at the same ratio---suggesting that its weight
schedule becomes too aggressive when tail-class sample counts
are very small.
```

### After
```
CB Loss underperforms CE specifically under severe imbalance
(IR=100)---degrading to F1-Macro of 0.406 against CE's 0.431
on CIFAR-10-LT, and 0.158 against 0.179 on CIFAR-100-LT---
while remaining competitive or superior to CE at moderate
imbalance (IR=10, 50), suggesting its weight schedule becomes
too aggressive only when tail-class sample counts are
critically small.
```

### Evidence
| Dataset | IR | CE F1 | CB F1 | CB > CE? |
|---------|----|-------|-------|----------|
| CIFAR-10-LT | 10 | 0.7103 | 0.7296 | ✅ |
| CIFAR-10-LT | 50 | 0.5203 | 0.5512 | ✅ |
| CIFAR-10-LT | 100 | 0.4306 | 0.4063 | ❌ |
| CIFAR-100-LT | 10 | 0.3513 | 0.3598 | ✅ |
| CIFAR-100-LT | 50 | 0.2101 | 0.1960 | ❌ |
| CIFAR-100-LT | 100 | 0.1786 | 0.1577 | ❌ |

CB Loss is only consistently below CE at IR=100. "Consistently underperforms" is inaccurate.

---

## Revision 3 — CIFAR-10-LT IR=100 Result Sentence

### Before
```
On CIFAR-10-LT at IR = 100, PLWCE achieves F1-Macro of
$0.458\pm0.018$, surpassing CE ($0.431\pm0.021$) and
CB Loss ($0.406\pm0.015$).
```

### After
```
On CIFAR-10-LT at IR = 100, PLWCE achieves the highest
F1-Macro of $0.458\pm0.018$, marginally ahead of Focal
($0.457\pm0.014$) and PWCE ($0.456\pm0.023$), and
substantially above CB Loss ($0.406\pm0.015$), which is
the only method to fall below CE ($0.431\pm0.021$).
```

### Evidence
Full ranking at CIFAR-10-LT IR=100 (mean F1-Macro, 5 seeds):

| Rank | Loss | F1-Macro | Δ vs CE |
|------|------|----------|---------|
| 1 | **PLWCE** | 0.4583 ± 0.0175 | +0.0277 |
| 2 | Focal | 0.4565 ± 0.0135 | +0.0259 |
| 3 | PWCE | 0.4564 ± 0.0233 | +0.0258 |
| 4 | LWCE | 0.4459 ± 0.0159 | +0.0153 |
| 5 | CE | 0.4306 ± 0.0209 | — |
| 6 | CB | 0.4063 ± 0.0150 | −0.0243 |

PWCE and Focal are within 0.002 of PLWCE. Omitting them overstates PLWCE's margin and misrepresents the competitive landscape.

---

## Revision 4 — Result Summary Sentence

### Before
```
Results demonstrate that LWCE and PLWCE consistently improve
over CE on F1-Macro and Few-class accuracy as the imbalance
ratio increases, while avoiding the performance degradation
exhibited by CB Loss and Focal Loss at high IRs.
```

### After
```
Results demonstrate that LWCE and PLWCE consistently improve
over CE on F1-Macro and Few-class accuracy as the imbalance
ratio increases, while CB Loss is the only method to degrade
below CE at severe imbalance across both benchmarks. Focal
Loss remains competitive on CIFAR-10-LT but fails to improve
over CE on CIFAR-100-LT at IR=100, highlighting the advantage
of logarithmic weighting in high-class-count settings.
```

### Evidence
| Dataset | IR | LWCE F1 | PLWCE F1 | CE F1 | Focal F1 | CB F1 |
|---------|----|---------|----------|-------|----------|-------|
| CIFAR-10-LT | 100 | 0.4459 | 0.4583 | 0.4306 | 0.4565 | 0.4063 |
| CIFAR-100-LT | 100 | 0.1846 | 0.1859 | 0.1786 | 0.1689 | 0.1577 |

Framing "CB Loss and Focal Loss" as a single group contradicts the diverging patterns between the two datasets.

---

## Revision 5 — Contribution Item 3

### Before
```
revealing that logarithmic re-weighting offers a more stable
and consistent improvement over CE than CB Loss or Focal Loss,
particularly in the high-imbalance regime.
```

### After
```
revealing that logarithmic re-weighting offers more stable and
consistent improvement over CE than CB Loss across both
benchmarks, and outperforms Focal Loss particularly on
CIFAR-100-LT where the number of tail classes is large.
```

### Evidence
| Dataset | LWCE vs Focal | Direction |
|---------|---------------|-----------|
| CIFAR-10-LT IR=100 | 0.4459 vs 0.4565 | Focal wins |
| CIFAR-100-LT IR=100 | 0.1846 vs 0.1689 | LWCE wins |

Claiming LWCE/PLWCE is broadly superior to Focal is overstated. The advantage of logarithmic weighting over Focal is specific to CIFAR-100-LT (100 classes, more severe tail), which is itself a meaningful and publishable finding.

---

## Key Experimental Data (Reference)

### CIFAR-10-LT — F1-Macro (mean ± std, n=5)
| Loss | IR=10 | IR=50 | IR=100 |
|------|-------|-------|--------|
| CE | 0.7103 ± 0.0074 | 0.5203 ± 0.0140 | 0.4306 ± 0.0209 |
| PWCE | 0.7235 ± 0.0092 | 0.5652 ± 0.0058 | 0.4564 ± 0.0233 |
| LWCE | 0.7182 ± 0.0100 | 0.5612 ± 0.0227 | 0.4459 ± 0.0159 |
| PLWCE | 0.7268 ± 0.0154 | 0.5551 ± 0.0209 | 0.4583 ± 0.0175 |
| CB | 0.7296 ± 0.0104 | 0.5512 ± 0.0167 | 0.4063 ± 0.0150 |
| Focal | 0.7143 ± 0.0160 | 0.5137 ± 0.0279 | 0.4565 ± 0.0135 |

### CIFAR-100-LT — F1-Macro (mean ± std, n=5)
| Loss | IR=10 | IR=50 | IR=100 |
|------|-------|-------|--------|
| CE | 0.3513 ± 0.0175 | 0.2101 ± 0.0058 | 0.1786 ± 0.0033 |
| PWCE | 0.3652 ± 0.0073 | 0.2107 ± 0.0061 | 0.1783 ± 0.0045 |
| LWCE | 0.3590 ± 0.0074 | 0.2173 ± 0.0076 | 0.1846 ± 0.0051 |
| PLWCE | 0.3582 ± 0.0246 | 0.2186 ± 0.0054 | 0.1859 ± 0.0088 |
| CB | 0.3598 ± 0.0155 | 0.1960 ± 0.0051 | 0.1577 ± 0.0041 |
| Focal | 0.3625 ± 0.0159 | 0.2084 ± 0.0062 | 0.1689 ± 0.0043 |

---

*Last updated: 2026-05-24*
