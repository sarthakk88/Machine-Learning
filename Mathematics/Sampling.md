# Sampling Techniques & Best Practices for Machine Learning

A reference guide to the **sampling strategies, resampling procedures, and data-splitting methods** required for reliable machine-learning workflows.

---

## Table of Contents
1. [Why Sampling Matters](#why-sampling-matters)
2. [Probability Sampling Techniques](#probability-sampling-techniques)
3. [Non-Probability Sampling Techniques](#non-probability-sampling-techniques)
4. [Sample-Size Determination & Statistical Power](#sample-size-determination--statistical-power)
5. [Sampling Bias & How to Mitigate It](#sampling-bias--how-to-mitigate-it)
6. [Resampling & Data-Splitting in ML](#resampling--data-splitting-in-ml)
7. [Handling Imbalanced Data](#handling-imbalanced-data)
8. [Python Snippets](#python-snippets)
9. [Best-Practice Checklist](#best-practice-checklist)

---

## Why Sampling Matters

Machine-learning models rarely have access to the entire population of data. Instead, they rely on a **sample** that must be:
1. **Representative** → captures diverse patterns present in the population.
2. **Unbiased** → avoids systematic errors introduced by the collection process.
3. **Sufficiently large** → provides adequate statistical power.

Poor sampling decisions propagate downstream, leading to *overfitting*, *biased estimates*, and *unfair predictions*.

---

## Probability Sampling Techniques

| Method | Workflow | When to Use | Pros | Cons |
|--------|----------|-------------|------|------|
| **Simple Random Sampling (SRS)** | Assign every unit an equal chance; select *n* at random. | Homogeneous populations; little prior info. | Easy; unbiased estimates. | May miss sub-groups; can be costly for large *N*. |
| **Stratified Sampling** | Split population into *strata* (e.g., gender); randomly sample within each stratum (proportional or disproportional). | Heterogeneous populations with known sub-groups. | Guarantees subgroup representation; ↑ precision. | Requires stratum info; complex weighting. |
| **Cluster Sampling** | Divide population into natural groups (clusters); randomly sample clusters; survey all (1-stage) or sample within (2-stage). | Geographically dispersed or costly populations. | ↓ cost/logistics. | ↑ intra-cluster correlation ⇒ larger SE; cluster selection bias. |
| **Systematic Sampling** | Select every *k*-th element after a random start. | Ordered lists (production lines). | Simple; evenly spread. | Risk of periodicity bias if order is patterned. |

**Notation**  
Population size = *N*; sample size = *n*; sampling fraction = *f = n/N*.

---

## Non-Probability Sampling Techniques

| Technique | Description | Typical Use-Cases |
|-----------|-------------|-------------------|
| **Convenience** | Select the easiest accessible units. | Quick pilots, exploratory analyses. |
| **Purposive/Judgmental** | Expert chooses units with desired traits. | Rare-disease studies, qualitative research. |
| **Quota** | Ensure sample hits quotas for key traits (age, gender) without randomization. | Market research. |
| **Snowball/Referral** | Existing subjects recruit future subjects. | Hard-to-reach networks, social graphs. |

*Warning*: Non-probability methods cannot support rigorous population inference without additional assumptions.

---

## Sample-Size Determination & Statistical Power

**Goal**:  
Choose $n$ large enough to detect a practically meaningful effect with high probability (**power ≥ 80%**) but small enough to be cost-efficient.

---

### Margin of Error (for Proportion Estimates)

**Formula**:  
$$
ME = Z_{\alpha/2} \sqrt{\frac{p(1-p)}{n}} \tag{1}
$$

Solve for $n$:  
$$
n = Z_{\alpha/2}^2 \cdot \frac{p(1-p)}{ME^2} \tag{2}
$$

Where:  
- $ME$ = desired margin of error (precision)  
- $p$ = estimated population proportion (use $p = 0.5$ for maximum variability when unknown)  
- $Z_{\alpha/2}$ = Z-score for desired confidence level (e.g., 1.96 for 95% CI)  
- $n$ = required sample size  

---

### Cochran’s Formula (for Large Populations)

When estimating proportions in large populations:  
$$
n_0 = \frac{Z_{\alpha/2}^2 \, p(1-p)}{e^2} \tag{3}
$$

For finite populations ($N$ total population size):  
$$
n = \frac{n_0}{1 + \frac{n_0 - 1}{N}} \tag{4}
$$

Where:  
- $n_0$ = initial sample size (large population assumption)  
- $N$ = finite population size  
- $e$ = desired margin of error (decimal form, e.g., 0.05 for 5%)  

---

### Notes on Statistical Power:
- **Power** = $1 - \beta$, where $\beta$ is the probability of Type II error (failing to reject a false null hypothesis).  
- Common choice: **80% power** (detect effect 4 out of 5 times if it exists).  
- Power depends on:
  - Effect size (larger effects need smaller $n$)
  - Significance level $\alpha$ (smaller $\alpha$ increases $n$)
  - Data variability (more variability → larger $n$)


### Power Analysis (Means)
Inputs: effect size *d*, significance α, desired power (1 − β). Use software (`power.t.test` in R, `statsmodels` in Python) to solve for *n*.

> **Rule of Thumb**: Larger variance or smaller effect sizes demand larger samples.

---

## Sampling Bias & How to Mitigate It

| Bias Type | Cause | Mitigation |
|-----------|-------|------------|
| **Selection** | Non-random inclusion/exclusion. | Probability sampling; weighting. |
| **Non-response** | Unit refuses or fails to respond. | Follow-ups, incentives, imputation. |
| **Under-coverage** | Frame misses part of population. | Frame audits; multiple frames. |
| **Survivorship** | Only surviving entities observed. | Track attrition; include defunct cases. |
| **Volunteer/Opt-in** | Self-selected participants. | Weighting adjustments; targeted outreach. |

Bias ≠ Variance trade-off: Eliminating bias may ↑ variance; aim for minimal *total* error.

---

## Resampling & Data-Splitting in ML

### 1 • Train / Validation / Test
Typical split: **80 % / 10 % / 10 %** (adjust per data volume).  
- **Training set** – model fitting.  
- **Validation set** – hyper-parameter tuning; early stopping.  
- **Test set** – unbiased final metric.

### 2 • k-Fold Cross-Validation
1. Shuffle & partition into *k* folds.  
2. Iterate: hold 1 fold as validation, train on remaining *k−1*.  
3. Average metrics.  
- Common *k*: 5 or 10.  
- **Stratified** CV preserves class ratios.

### 3 • Leave-One-Out (LOOCV)
Special case *k = N* – maximal data usage; high variance; expensive.

### 4 • Bootstrap Resampling
- Sample *n* observations **with replacement**; keep OOB (≈36 %) for testing.  
- Repeat *B* times (≥1000).  
- Average to estimate model skill distribution; supports confidence intervals.

### 5 • Monte-Carlo (Repeated Random Splits)
Randomly split into train/test many times; aggregate.

---

## Handling Imbalanced Data

| Strategy | Idea | Typical Tools |
|----------|------|--------------|
| **Random Over-Sampling** | Duplicate minority samples. | `RandomOverSampler` (imblearn) |
| **SMOTE / ADASYN** | Synthesize new minority points. | `SMOTE`, `SMOTE-Tomek`, `ADASYN` |
| **Random Under-Sampling** | Remove majority samples. | `RandomUnderSampler` |
| **Hybrid (SMOTEENN, SMOTETomek)** | Over-sample + Clean noise. | `SMOTEENN` |
| **Class Weighting** | Adjust loss weights inversely to class frequency. | `class_weight` arg (Keras, sklearn) |

*Always evaluate with stratified CV & metrics robust to imbalance (ROC-AUC, F1, PR-AUC).* 

---

## Python Snippets

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
import numpy as np

# 1. Simple train/validation/test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, stratify=y_temp, random_state=42)  # 0.111 ≈ 0.1 overall

# 2. Stratified 5-fold CV iterator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (idx_train, idx_val) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}: train={len(idx_train)}, val={len(idx_val)})")

# 3. Bootstrap sample with OOB set
boot_idx = resample(np.arange(len(X)), replace=True, n_samples=len(X))
oob_idx  = np.setdiff1d(np.arange(len(X)), boot_idx)
X_boot, y_boot = X[boot_idx], y[boot_idx]
X_oob,  y_oob  = X[oob_idx],  y[oob_idx]
```

---

## Best-Practice Checklist

- [ ] **Define the target population** and create an up-to-date sampling frame.
- [ ] **Choose an appropriate sampling design** (probability > non-probability for inference).
- [ ] **Estimate required sample size** via power analysis or margin-of-error formulas.
- [ ] **Document the selection process** and any adjustments (weights, post-stratification).
- [ ] **Assess representativeness**; compare sample demographics to population.
- [ ] **Detect & mitigate sampling bias** (selection, non-response, survivorship).
- [ ] **Use stratified CV or bootstrap** to obtain stable performance estimates.
- [ ] **For imbalanced problems**, apply resampling or cost-sensitive methods *inside* CV folds.
- [ ] **Report uncertainty** (confidence intervals) around metrics.

> **Remember:** *Garbage in, garbage out.* Sound sampling is the foundation of trustworthy machine-learning models.
