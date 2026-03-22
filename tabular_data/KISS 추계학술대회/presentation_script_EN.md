# Presentation Script (English)
# Logarithmic Scale Weighted Cross-Entropy for Class-Imbalanced Learning
# KIIS 2025 Fall Conference

---

## Slide 1 — Title

Good morning, everyone. My name is Seung Ho Go, from the Department of Software at Sejong University. Today, I will be presenting our work on a new loss function called **Logarithmic Scale Weighted Cross-Entropy**, or LWCE, designed to address the class imbalance problem. This research was motivated by the numerical instability inherent in conventional weighting-based loss functions, and we propose a log-scaling approach as a simple yet effective remedy.

---

## Slide 2 — Table of Contents

Here is the roadmap for today's talk. I will start with the **research motivation and related work**, then introduce our **proposed methodology**, followed by the **experimental results**, and close with a **discussion and future directions**.

---

## Slide 3 — Latent Majority Bias in Imbalanced Data

Let me begin with why this problem matters.

In many real-world domains — healthcare, finance, manufacturing — the number of samples per class is far from balanced. When a model is trained on such data, the majority class dominates the learning process, and the model ends up simply ignoring the minority class.

Consider the example on the right. This model achieves **99% accuracy** — which sounds impressive. But look at the confusion matrix: it predicted every single patient as healthy. It detected **zero cancer cases**. The reason accuracy looks so high is that 99% of the data is already healthy samples. This is what we call *deceptive accuracy*, and in a medical context, missing every positive case can be fatal.

This is the core problem we set out to solve.

---

## Slide 4 — Common Approaches to Class Imbalance

There are two broad families of solutions for class imbalance.

The first is **data resampling** — directly modifying the training set. This includes oversampling methods like SMOTE and ADASYN, and undersampling methods like Tomek Links and ENN. However, resampling has a fundamental risk: oversampling can introduce artificial samples that lead to overfitting, while undersampling discards real data and distorts the true distribution.

The second family is **cost-sensitive learning** — rather than changing the data, we change the algorithm's behavior. By assigning a higher penalty to misclassifying minority samples, we let the model stay aware of their importance without touching the data.

Among cost-sensitive methods, the Standard Cross-Entropy loss has a well-known limitation: the total loss is dominated by the majority class, and the gradient signal from the minority class gets overwhelmed. This is why various reweighted loss functions have been proposed — and that is where our work sits.

---

## Slide 5 — Existing Methods (I): Inverse Weighting Schemes

The most straightforward cost-sensitive approach is **Weighted Cross-Entropy**, or WCE. The idea is simple: assign a higher weight to classes with fewer samples, so the loss penalizes their misclassification more.

The standard WCE weight is inversely proportional to class frequency — `1/n_i`. A softer variant takes the square root, `1/sqrt(n_i)`, to reduce the aggressiveness.

These methods are intuitive, but they share a critical flaw: **when a minority class has very few samples, the weight explodes**. In extreme imbalance scenarios — say, fraud detection with a 587:1 ratio — `1/n_i` can produce enormous weights that make training numerically unstable and prone to severe overfitting.

---

## Slide 6 — Existing Methods (II): Advanced Losses

Beyond simple inverse weighting, more sophisticated approaches have been proposed.

**Focal Loss** shifts the focus from *class frequency* to *sample difficulty*. Rather than applying a fixed weight per class, it dynamically down-weights the loss contribution of easy samples — those the model already classifies with high confidence — and concentrates learning on hard, ambiguous samples. The hyperparameter gamma controls how aggressively this focusing happens, while alpha provides a static per-class balance factor.

**Class-Balanced Loss** takes a different angle. It observes that as you add more samples to a class, the marginal gain in information decreases — data overlaps. So instead of raw sample counts, it introduces the concept of the *effective number of samples*, using a hyperparameter beta to discount redundancy. This gives a more principled estimate of how much information each class actually contributes.

Both methods represent meaningful advances, but they require careful hyperparameter tuning, and in some extreme-imbalance scenarios, numerical instability can still emerge.

---

## Slide 7 — (Transition to Methodology)

Now let me introduce our proposed solution.

---

## Slide 8 — Proposed Methodology (I): The Core Idea — LWCE

The root cause of WCE's instability is the `1/n_i` term. When `n_i` is small, this term grows without bound.

Our key insight is straightforward: **replace `n_i` with `log(n_i + 1)`**.

This is the essence of **Log-Weighted Cross-Entropy**, or LWCE. The logarithm compresses the weight scale dramatically. Look at the graph on the right: the standard WCE weight shoots up to 10,000 when the minority class has only one sample. Our LWCE weight, by contrast, stays below about 15 across the entire range — completely stable.

By applying log-scaling, we preserve the core intuition of WCE — give higher weight to rare classes — while fundamentally taming the explosion. The model can focus on the minority class without being destabilized by extreme weights.

---

## Slide 9 — Proposed Methodology (II): Power-Scaled LWCE (PLWCE)

While LWCE alone is already a strong solution, we wanted to give practitioners more flexibility. So we introduce an exponent parameter **alpha** to control the focusing strength. We call this generalization **Power-Scaled LWCE**, or PLWCE.

The weight becomes: `1 / (log(n_i + 1))^alpha`

Think of alpha as a **sensitivity dial**:
- When **alpha = 1.0**, you get the standard LWCE.
- When **alpha > 1.0**, the weights become more aggressive — useful when imbalance is severe and you need to push the model harder toward the minority class.
- When **alpha < 1.0**, the weights become more conservative — for cases where even log-scaling provides more focus than needed.

As shown in the graph, increasing alpha amplifies the relative weight assigned to minority classes in a smooth, controlled manner — unlike WCE, which would diverge under the same conditions.

PLWCE essentially generalizes LWCE and provides an optional tuning handle for achieving peak performance in critical scenarios.

---

## Slide 10 — (Transition to Experiments)

Now let's look at how these methods perform in practice.

---

## Slide 11 — Experimental Setup (I): Datasets & Model

We evaluate our methods on **11 benchmark datasets in total** — 7 binary classification datasets and 4 multi-class datasets. The imbalance ratios span from a relatively mild **2.3:1** (German Credit) all the way to **587:1** (Credit Card Fraud), covering a wide range of real-world imbalance conditions.

For the base model, we use **XGBoost with a linear booster**, `gblinear`. This is a strong, reproducible baseline that handles both binary and multi-class tasks efficiently. Using a linear model also ensures that differences in performance come from the loss function, not from the complexity of the model itself.

---

## Slide 12 — Experimental Setup (II): Metrics & Optimization

For imbalanced data, standard accuracy is misleading — as we saw on the first example. We therefore report two metrics:

- **F1-Score**: the harmonic mean of precision and recall, averaged across classes. It explicitly penalizes methods that ignore minority classes.
- **PR-AUC**: the Area Under the Precision-Recall curve. This metric is especially sensitive to minority-class performance and captures fine-grained ranking quality.

For fair comparison, we use **Optuna** with **3-fold cross-validation and 100 trials per dataset** for each method. The search space includes the learning rate, number of estimators, L1 and L2 regularization — as well as the loss-specific parameters: gamma for Focal, beta for CB, and alpha for PLWCE.

This ensures that every method gets the best possible set of hyperparameters, making the comparison fair.

---

## Slide 13 — Full Experimental Results (I): Binary Datasets

Here are the results on the six binary classification datasets, ordered by imbalance ratio. For each dataset, we color the **1st-place result in red** and 2nd and 3rd place in blue.

Our proposed LWCE and PLWCE consistently appear in the top rankings across these datasets. Notably, on **Credit Card Fraud** (IR = 587:1) — the most extreme case — PLWCE achieves the highest F1-Score of 0.806, outperforming all baselines including Focal Loss and CB Loss.

On the **APS Failure** and **SECOM** datasets, which represent more moderate but still significant imbalance, our methods again rank competitively. One thing to highlight is that conventional WCE sometimes degrades significantly — for example, its F1 on APS Failure drops to 0.648, the worst among all methods — which is a direct consequence of weight explosion under high imbalance.

---

## Slide 14 — Full Experimental Results (II): Multi-Class Datasets

Moving to the multi-class results, the picture is equally encouraging.

On the **Page Blocks** dataset, WCE's F1-Score collapses to 0.006 — essentially random performance — while LWCE and PLWCE maintain 0.815 and 0.834 respectively. This is a striking illustration of WCE's instability in multi-class settings, where multiple minority classes can each trigger extreme weights simultaneously.

Across the multi-class datasets as a whole — Page Blocks, Yeast, Steel Faults, and Glass — LWCE and PLWCE demonstrate stable, top-tier performance. Even in cases where standard CE achieves the best result on a specific metric, our methods are always close behind. We never see the catastrophic failure modes that plague WCE in these settings.

---

## Slide 15 — Overall Performance Summary: Heatmap

To summarize performance across all 11 datasets, we use a **Borda-count heatmap**. On the Y-axis, methods are ranked from best (top) to worst (bottom) by average rank. On the X-axis, datasets are ordered from left to right by increasing imbalance ratio.

The key finding is in the **top-right region** of each heatmap: the most severely imbalanced datasets. LWCE and PLWCE consistently show the deepest green colors there, meaning they rank 1st or 2nd most frequently on the hardest problems.

Looking at the overall rankings: **LWCE achieves an average rank of 3.73 on F1 and 3.09 on PR-AUC**, while **PLWCE scores 3.82 and 3.36** respectively. Both methods sit at the top of the Borda count. This confirms that our proposed methods are not only competitive but robustly superior — especially where it counts most, on datasets with extreme imbalance.

---

## Slide 16 — (Transition to Discussion)

Now let me discuss what these results tell us, and where we go from here.

---

## Slide 17 — Discussion (I): The Benefit of Tuning Alpha

One question from our results is: *when should you bother tuning alpha?*

The answer is: LWCE with alpha = 1.0 is already a very strong baseline. In most of our 11 datasets, it performs at the top without any additional tuning. However, in **5 out of 11 datasets**, tuning alpha above 1.0 allowed PLWCE to find an even better optimum — improving both F1 and PR-AUC simultaneously.

We call this the **"Performance Booster"** effect. For standard cases, LWCE is sufficient. But for truly extreme imbalance scenarios where every percentage point matters — medical diagnosis, fraud detection — PLWCE's alpha dial gives practitioners a meaningful additional lever to squeeze out further gains.

---

## Slide 18 — Discussion (II): Difficulty of Isolating Alpha

Of course, we also have to be honest about the limitations.

Ideally, we would like to derive a clean mathematical relationship between the optimal alpha and the imbalance ratio. However, this turns out to be difficult in practice. Each dataset has its own optimal combination of learning rate, regularization strength, and other hyperparameters. When we tried to fix all other parameters across datasets and vary only alpha, the results became unreliable — the fixed parameters were simply too sub-optimal for some datasets.

As shown in the scatter plot, there is a **weak positive trend**: datasets with higher imbalance ratios tend to benefit from larger alpha values. But the overall regression line across all datasets does not show a clean monotonic relationship. The per-dataset variability is too large.

This remains an open challenge — and it motivates our future work.

---

## Slide 19 — Future Work (I): Model & Weighting Expansion

Looking ahead, there are two main directions for expanding this work.

The first is **model expansion**. Our current validation used a linear XGBoost model. We plan to extend this to tree-based non-linear models — standard XGBoost gbtree, LightGBM — and more importantly, to deep learning architectures: CNNs and Transformers. In particular, we are already exploring LWCE in the context of **medical image segmentation**, where class imbalance is severe and the stakes are high.

The second direction is combining our class-level static weighting with **dynamic sample-level weighting**. PLWCE currently assigns a fixed weight to each class based on its frequency. Focal Loss, on the other hand, dynamically adjusts the weight of each *sample* based on how confident the model already is. We believe a **hybrid loss** that combines PLWCE's stable class-level reweighting with Focal's per-sample focusing could address both the macro imbalance structure and the micro difficulty structure simultaneously.

---

## Slide 20 — Future Work (II): Automated Alpha Tuning

The third future direction directly addresses the limitation we just discussed: **automating alpha selection**.

Our plan is to design a **controlled experiment** using a large dataset — such as Credit Card Fraud — where we fix the minority class size and progressively subsample the majority class to create a series of datasets with known, controlled imbalance ratios: say, 5:1, 10:1, 50:1, 100:1, 500:1.

By running alpha-only tuning in these controlled settings — where all other hyperparameters are held constant — we can isolate the alpha-IR relationship cleanly. The ultimate goal is to derive a **heuristic or formula** that allows users to set alpha automatically from the data's imbalance ratio, eliminating the need for expensive trial-and-error tuning.

---

## Slide 21 — Conclusion

To summarize our contributions:

**First**, we identified and formally highlighted the **numerical instability** of standard Weighted Cross-Entropy under extreme class imbalance.

**Second**, we proposed **LWCE** — a simple, parameter-free solution that replaces the unstable `1/n_i` weighting with `1/log(n_i + 1)`, fundamentally preventing weight explosion.

**Third**, we generalized this into **PLWCE** by introducing the alpha exponent, which acts as an optional performance booster for the most challenging imbalanced datasets.

**Fourth**, we validated both methods across **11 diverse benchmark datasets**, demonstrating that LWCE and PLWCE consistently rank at the top — particularly on datasets with severe imbalance, where stability and discriminative power matter most.

We believe LWCE and PLWCE offer a practical, principled, and widely applicable solution to class-imbalanced learning.

---

## Slide 22 — References

*(Slide shown — no verbal script needed, or briefly say:)*

The key references supporting this work include the original Focal Loss paper by Lin et al., the Class-Balanced Loss paper by Cui et al., and prior work on weighted cross-entropy by Phan & Yamamoto and Wang et al. Full citations are shown on this slide.

---

## Slide 23 — Thank You

That concludes my presentation. Thank you very much for your attention. I am happy to take any questions.

---

## Q&A Preparation Notes

**Q: Why use a linear XGBoost model instead of a tree model?**
> A linear model has fewer confounding hyperparameters, making it easier to isolate the effect of the loss function. We plan to extend to gbtree in future work.

**Q: How does LWCE compare to Focal Loss conceptually?**
> Focal Loss focuses on *sample difficulty* (easy vs. hard samples) using a dynamic per-sample weight. LWCE focuses on *class frequency* using a static per-class weight. They are complementary — Focal controls the micro difficulty structure, LWCE controls the macro class imbalance. Combining them is one of our future directions.

**Q: Is there a recommended default value for alpha in PLWCE?**
> We recommend starting with alpha = 1.0, which gives standard LWCE. If your imbalance ratio is above ~50:1 and performance on the minority class is still insufficient, try alpha in the range of 1.5 to 3.0.

**Q: Does LWCE work for multi-class settings?**
> Yes — as shown in the results, LWCE and PLWCE handle multi-class imbalance significantly better than WCE, which can catastrophically collapse in that setting. The log-scaling stabilizes all class weights simultaneously.
