# Presentation Script (English)
# Logarithmic Scale Weighted Cross-Entropy for Class-Imbalanced Learning
# KIIS 2025 Fall Conference

---

## Slide 1 — Title

Good morning. I am Seung Ho Go from the Department of Software at Sejong University. Today I will present our work on a new loss function called **Logarithmic Scale Weighted Cross-Entropy**, or LWCE, designed to address class imbalance. This research was motivated by the numerical instability inherent in conventional weighting-based loss functions, and we propose a log-scaling approach as a simple yet effective remedy.

---

## Slide 2 — Table of Contents

Today's talk is organized as follows: research motivation and related work, proposed methodology, experimental results, and discussion with future directions.

---

## Slide 3 — Latent Majority Bias in Imbalanced Data

In real-world domains such as healthcare and finance, class imbalance is a common challenge. When a model is trained on imbalanced data, the majority class dominates the learning process and the minority class is effectively ignored.

Consider the example on the right. The model achieves 99% accuracy, yet it fails to detect a single cancer case — predicting every patient as healthy. This is because 99% of the data is already the healthy class. This kind of deceptive accuracy can be fatal in medical contexts, and it is precisely the problem we set out to solve.

---

## Slide 4 — Common Approaches to Class Imbalance

Two broad families of solutions exist for class imbalance.

The first is **data resampling** — directly modifying the training set through oversampling or undersampling. However, oversampling risks overfitting through artificial sample generation, and undersampling discards real data and distorts the true distribution.

The second is **cost-sensitive learning**, which addresses the problem at the algorithmic level without altering the data. Standard cross-entropy has a fundamental limitation here: the total loss is dominated by the majority class, suppressing the gradient signal from minority classes. This has motivated a range of reweighted loss functions — and that is where our work sits.

---

## Slide 5 — Existing Methods (I): Inverse Weighting

The most intuitive cost-sensitive approach is **Weighted Cross-Entropy (WCE)**, which assigns higher weights to classes with fewer samples using the inverse frequency `1/n_i`. A softer variant applies a square root, `1/sqrt(n_i)`, to moderate aggressiveness.

However, these methods share a critical flaw: **as the minority class becomes smaller, the weight grows without bound**, producing numerical instability and severe overfitting in extreme imbalance scenarios.

---

## Slide 6 — Existing Methods (II): Advanced Losses

Beyond simple inverse weighting, more sophisticated approaches have been proposed.

**Focal Loss** shifts focus from class frequency to *sample difficulty*. It dynamically down-weights easy samples — those classified with high confidence — and concentrates learning on hard, ambiguous ones. The gamma parameter controls this focusing effect, while alpha provides a static per-class balance factor.

**Class-Balanced Loss** introduces the concept of *effective number of samples*, accounting for data redundancy that grows with sample count. This yields a more principled weighting scheme reflecting actual information content per class.

Both methods represent meaningful advances, yet they require careful hyperparameter tuning and can still exhibit numerical instability in extreme settings.

---

## Slide 7 — (Transition to Methodology)

Now let me introduce our proposed solution.

---

## Slide 8 — Proposed Methodology (I): LWCE

The root cause of WCE's instability is the `1/n_i` term, which grows without bound as `n_i` decreases.

Our key insight is to **replace `n_i` with `log(n_i + 1)`**. This is the essence of **Log-Weighted Cross-Entropy (LWCE)**. As shown in the graph, the standard WCE weight explodes to 10,000 when the minority class has only a single sample. Our LWCE weight, by contrast, remains stable across the entire range — below approximately 15.

By applying log-scaling, we preserve the core intuition of WCE — assigning higher weight to rare classes — while fundamentally preventing weight explosion.

---

## Slide 9 — Proposed Methodology (II): PLWCE

To provide additional flexibility, we introduce an exponent parameter **alpha**, generalizing LWCE into **Power-Scaled LWCE (PLWCE)**.

The weight becomes: `1 / (log(n_i + 1))^alpha`

Alpha acts as a **sensitivity dial**: alpha = 1.0 recovers standard LWCE; alpha > 1.0 amplifies focus on the minority class for severe imbalance; alpha < 1.0 applies more conservative weighting. As shown in the graph, increasing alpha raises minority class weights in a smooth, bounded manner — unlike WCE, which diverges under equivalent conditions.

---

## Slide 10 — (Transition to Experiments)

Now let us look at how these methods perform in practice.

---

## Slide 11 — Experimental Setup (I): Datasets & Model

We evaluate on **11 benchmark datasets** — 7 binary and 4 multi-class — with imbalance ratios ranging from 2.3:1 to 587:1. The base model is **XGBoost with a linear booster (gblinear)**, a strong and reproducible baseline that allows differences in performance to be attributed to the loss function rather than model complexity.

---

## Slide 12 — Experimental Setup (II): Metrics & Optimization

Given the limitations of accuracy on imbalanced data, we report **F1-Score** and **PR-AUC** — metrics that explicitly capture minority-class performance.

For fair comparison, we use **Optuna** with **3-fold cross-validation and 100 trials per dataset** for all methods, optimizing over learning rate, regularization, and loss-specific parameters (gamma, beta, or alpha).

---

## Slide 13 — Results (I): Binary Datasets

Here are the results on six binary datasets, ordered by imbalance ratio. First-place results are highlighted in red; second and third in blue.

Our proposed methods consistently appear at the top across these datasets. Notably, on **Credit Card Fraud** (IR = 587:1), PLWCE achieves the highest F1-Score, outperforming all baselines. In contrast, WCE sometimes degrades severely — for example, its F1 on APS Failure drops to 0.648, the lowest among all methods — a direct consequence of weight explosion under high imbalance.

---

## Slide 14 — Results (II): Multi-Class Datasets

On the multi-class datasets, the results are equally encouraging. On **Page Blocks**, WCE's F1-Score collapses to 0.006 — near-random performance — while LWCE and PLWCE maintain 0.815 and 0.834 respectively. This strikingly illustrates WCE's instability when multiple minority classes simultaneously trigger extreme weights.

Across all four multi-class datasets, LWCE and PLWCE demonstrate stable, top-tier performance. Even when standard CE achieves the best result on a specific metric, our methods remain consistently competitive — without the catastrophic failures observed with WCE.

---

## Slide 15 — Overall Summary: Borda-Count Heatmap

To summarize across all 11 datasets, we use a **Borda-count heatmap**. Methods are ranked top-to-bottom by average rank; datasets are ordered left-to-right by increasing imbalance ratio.

The key observation is in the **top-right region** — the most severely imbalanced datasets. LWCE and PLWCE consistently show the deepest green, indicating top-1 or top-2 rankings on the hardest problems. Overall, **LWCE achieves average ranks of 3.73 (F1) and 3.09 (PR-AUC)**, with PLWCE scoring 3.82 and 3.36 respectively — confirming robustly superior performance where it matters most.

---

## Slide 16 — (Transition to Discussion)

Let me now discuss what these results tell us and where we go from here.

---

## Slide 17 — Discussion (I): The Benefit of Tuning Alpha

LWCE at alpha = 1.0 is already a strong baseline for most datasets. However, in **5 out of 11 datasets**, tuning alpha above 1.0 allowed PLWCE to improve both F1 and PR-AUC simultaneously — a **"performance booster"** effect for the most challenging cases. For standard settings, LWCE suffices; for extreme imbalance scenarios where every percentage point matters, PLWCE's alpha dial provides a meaningful additional lever.

---

## Slide 18 — Discussion (II): Difficulty of Isolating Alpha

We must also acknowledge a limitation. Ideally, we would derive a clean relationship between the optimal alpha and the imbalance ratio. In practice, however, each dataset has its own optimal combination of learning rate, regularization, and other hyperparameters, making it difficult to isolate the effect of alpha alone.

As shown in the scatter plot, there is a **weak positive trend** for datasets with IR ≥ 10, but the overall regression line does not show a clean monotonic relationship. This remains an open challenge and directly motivates our future work.

---

## Slide 19 — Future Work (I): Model Expansion & Hybrid Loss

Two main directions remain.

The first is **model expansion**. Having validated LWCE on a linear model, we plan to extend to tree-based non-linear models (XGBoost gbtree, LightGBM) and deep learning architectures (CNNs, Transformers), with particular focus on **medical image segmentation** where class imbalance is severe and clinically consequential.

The second is developing a **hybrid loss** that combines PLWCE's stable class-level reweighting with Focal Loss's per-sample dynamic weighting. Such a method would simultaneously address macro imbalance structure and micro sample-difficulty structure.

---

## Slide 20 — Future Work (II): Automated Alpha Tuning

The third direction directly addresses the limitation just described. We plan a **controlled experiment** using a large dataset — such as Credit Card Fraud — where the minority class size is fixed and the majority class is progressively subsampled to construct datasets with known imbalance ratios (5:1, 10:1, 50:1, 100:1, 500:1).

By running alpha-only tuning in this controlled setting, we aim to isolate the alpha–IR relationship and ultimately derive a **heuristic or formula** that enables automatic alpha selection from data characteristics alone — eliminating the need for expensive trial-and-error tuning.

---

## Slide 21 — Conclusion

To summarize our contributions:

**First**, we formally identified the numerical instability of standard Weighted Cross-Entropy under extreme class imbalance.

**Second**, we proposed **LWCE**, which replaces the unstable `1/n_i` weighting with `1/log(n_i + 1)`, fundamentally preventing weight explosion with no additional parameters.

**Third**, we generalized this into **PLWCE** via the alpha exponent, offering an optional performance booster for the most challenging imbalanced settings.

**Fourth**, we validated both methods across **11 diverse benchmark datasets**, demonstrating consistently top-ranked performance — particularly on datasets with severe imbalance.

We believe LWCE and PLWCE offer a practical, principled, and widely applicable solution to class-imbalanced learning.

---

## Slide 22 — References

*(Slide shown — briefly acknowledge key references)*

Key references include the Focal Loss paper by Lin et al., the Class-Balanced Loss paper by Cui et al., and prior work on weighted cross-entropy by Phan & Yamamoto and Wang et al.

---

## Slide 23 — Thank You

That concludes my presentation. Thank you for your attention. I am happy to take any questions.

---

## Q&A Preparation Notes

**Q: Why use a linear XGBoost model instead of a tree model?**
> A linear model minimizes confounding hyperparameters, making it easier to isolate the effect of the loss function. Extension to gbtree is planned.

**Q: How does LWCE compare to Focal Loss?**
> Focal Loss addresses *sample difficulty* via dynamic per-sample weighting. LWCE addresses *class frequency* via static per-class weighting. They are complementary — combining them is a planned future direction.

**Q: Is there a recommended default for alpha in PLWCE?**
> Start with alpha = 1.0 (standard LWCE). For imbalance ratios above ~50:1 where minority performance is still insufficient, try alpha in the range of 1.5 to 3.0.

**Q: Does LWCE work for multi-class settings?**
> Yes — as demonstrated in the results, LWCE and PLWCE handle multi-class imbalance significantly better than WCE, which catastrophically collapses in that setting.
