# Statistics & Probability Notes for Machine Learning

## 1. Foundation Concepts

| Concept | Key Points | Machine Learning Relevance |
|---------|------------|---------------------------|
| **Population vs. Sample** | Population = entire set; sample = subset drawn by a **probability sampling** method such as simple, stratified, cluster, or systematic sampling. | Models train on samples; good sampling prevents bias propagation. |
| **Descriptive vs. Inferential Statistics** | Descriptive condenses data (mean, median, mode, variance); inferential draws conclusions about the population (confidence intervals, hypothesis tests). | Descriptive stats guide preprocessing; inferential stats justify feature or model choices. |

---

## 2. Descriptive Statistics

### 2.1 Measures of Central Tendency

- **Mean**: Arithmetic average—sensitive to outliers
  - Formula: $\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$
  - Use cases: Model evaluation, feature normalization

- **Median**: Middle value—robust to skewed data
  - More reliable for non-normal distributions
  - Use cases: Data preprocessing, outlier detection

- **Mode**: Most frequent value—useful for categorical predictors
  - Can be multimodal (multiple modes)
  - Use cases: Classification problems, categorical analysis

### 2.2 Measures of Dispersion

- **Range**: Difference between max and min values

- **Interquartile Range (IQR)**: Range between 25th and 75th percentiles

- **Variance**: Average squared deviation from the mean  
  - **Population Variance**:  
    $\sigma^2 = \frac{\sum_{i=1}^{N}(x_i - \mu)^2}{N}$  
  - **Sample Variance**:  
    $s^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n - 1}$

- **Standard Deviation**: Square root of variance  
  - **Population Standard Deviation**:  
    $\sigma = \sqrt{\sigma^2}$  
  - **Sample Standard Deviation**:  
    $s = \sqrt{s^2}$  
  - Critical for feature scaling and normalization

### 2.3 Shape of Distribution

#### **Skewness** – Asymmetry Measure

Skewness means **lack of symmetry**. A distribution is symmetric or normal when frequencies are symmetrically distributed about the mean. For a symmetric distribution: **mean = mode = median**.

- **Positive (Right) Skewed**: Tail extends to the right
  - **mean > median > mode**
  - Longer or fatter tail on the right side

- **Negative (Left) Skewed**: Tail extends to the left  
  - **mean < median < mode**
  - Longer or fatter tail on the left side

**Karl Pearson's Coefficient of Skewness** (First Formula):  
$Sk = \frac{\bar{x} - \text{Mode}}{s}$  

Where:  
- $\bar{x}$ = sample mean (use $\mu$ for population mean)  
- $\text{Mode}$ = most frequent value in the dataset  
- $s$ = sample standard deviation (use $\sigma$ for population)  

**Interpretation**:  
- $Sk > 0$: Positively skewed (right tail longer)  
- $Sk < 0$: Negatively skewed (left tail longer)  
- $Sk = 0$: Symmetric distribution

#### **Kurtosis** – Tailedness and Peakedness

Kurtosis describes the shape of a distribution's tails and peakedness compared to a normal distribution. It provides insights into outlier presence and concentration around the mean.

**Types of Kurtosis**:

1. **Mesokurtic (Normal Distribution)**:
   - Kurtosis value = 3
   - Moderate peak and tail behavior
   - Example: Standard normal distribution

2. **Leptokurtic (High Kurtosis)**:
   - Kurtosis > 3
   - Sharp peak and heavy tails
   - Frequent extreme values (many outliers)
   - Example: t-distribution with low degrees of freedom

3. **Platykurtic (Low Kurtosis)**:
   - Kurtosis < 3
   - Flatter peak and lighter tails
   - Example: Uniform distribution

### 2.4 Outlier Detection Methods

Identifying outliers is crucial for improving statistical model accuracy:

#### 1. **Using Descriptive Statistics (Z-Score)**  
- **Population Z-score**:  
  $z = \frac{x - \mu}{\sigma}$  

- **Sample Z-score**:  
  $z = \frac{x - \bar{x}}{s}$  

Where:  
- $x$ = data point  
- $\mu$, $\sigma$ = population mean and standard deviation  
- $\bar{x}$, $s$ = sample mean and standard deviation  

Outliers typically have $|z| > 3$.

#### 2. **Using Interquartile Range (IQR) Method**  
$IQR = Q_3 - Q_1$  

Outlier criteria:  
- $x < Q_1 - 1.5 \times IQR$  
- $x > Q_3 + 1.5 \times IQR$  

#### 3. **Visualization Methods**
- **Box Plot**: Outliers appear as individual points outside whiskers
- **Histogram**: Identifies extreme values in distribution
- **Scatter Plot**: Detects unusual data points in bivariate data

#### 4. **Machine Learning-Based Approaches**
- **Isolation Forest**: Anomaly detection using decision trees
- **DBSCAN**: Identifies low-density points as outliers
- **LOF (Local Outlier Factor)**: Measures isolation compared to neighbors

---

## 3. Probability Essentials

### 3.1 Basic Terminology

- **Sample Space** ($\Omega$): Set of all possible outcomes.  
- **Event** ($A$): Subset of the sample space.  
- **Probability**:  
  $$
  P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}}
  $$

**Key Probability Rules**:

- **Addition Rule**:  
  $$
  P(A \cup B) = P(A) + P(B) - P(A \cap B)
  $$

- **Multiplication Rule**:  
  $$
  P(A \cap B) = P(A) \cdot P(B|A)
  $$

- **Conditional Probability**:  
  $$
  P(A|B) = \frac{P(A \cap B)}{P(B)}
  $$

### 3.2 Bayes' Theorem

Bayes’ theorem updates the probability of an event based on new information:  

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

Where:  
- $P(A|B)$ = Posterior probability (probability of $A$ given $B$)  
- $P(B|A)$ = Likelihood (probability of $B$ given $A$)  
- $P(A)$ = Prior probability of $A$  
- $P(B)$ = Marginal probability of $B$

**Applications in ML**:
- Naive Bayes classifiers
- Bayesian optimization
- Medical diagnosis systems
- Spam filtering

### 3.3 Common Probability Distributions

#### **Discrete Distributions**

- **Bernoulli Distribution**: Single trial with binary outcome.  
  - Parameter: $p$ (probability of success)  
  - Used in: Binary classification

- **Binomial Distribution**: Multiple independent Bernoulli trials.  
  - Formula: $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$  
  - Applications: A/B testing, quality control

- **Poisson Distribution**: Events occurring in fixed intervals.  
  - Formula: $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$  
  - Used in: Recommendation systems, fraud detection

- **Geometric Distribution**: Number of trials until the first success.  
  - Formula: $P(X = k) = (1-p)^{k-1} p$  
  - Used in: Reliability testing, modeling time to first event

- **Negative Binomial Distribution**: Number of trials until a fixed number of successes.  
  - Formula: $P(X = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}$  
  - Used in: Overdispersed count data, modeling repeated events

- **Hypergeometric Distribution**: Probability of $k$ successes in $n$ draws **without** replacement from a finite population.  
  - Formula:  
    $$
    P(X = k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}
    $$  
  - Used in: Quality control, sampling without replacement

- **Multinomial Distribution**: Generalization of binomial to more than two outcomes.  
  - Formula:  
    $$
    P(X_1 = x_1, \dots, X_k = x_k) = \frac{n!}{x_1! \dots x_k!} p_1^{x_1} \dots p_k^{x_k}
    $$  
  - Used in: Text classification (word counts), categorical outcome modeling

#### **Continuous Distributions**

- **Normal Distribution**: Bell-shaped curve.  
  - Formula:  
    $$
    f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    $$
  - Central to many ML algorithms and the Central Limit Theorem.

- **Standard Normal Distribution**: Special case when $\mu = 0$, $\sigma = 1$.  
  - **Z-score transformation**:  
    $$
    z = \frac{x - \mu}{\sigma}
    $$
  - Represents how many standard deviations $x$ is from the mean.

- **Uniform Distribution**: All outcomes equally likely over an interval $[a,b]$.  
  - Formula:  
    $$
    f(x) = \frac{1}{b-a}, \quad a \le x \le b
    $$
  - Used in: Random sampling, parameter initialization.

- **Exponential Distribution**: Models time between events in a Poisson process.  
  - Formula:  
    $$
    f(x) = \lambda e^{-\lambda x}, \quad x \ge 0
    $$
  - Applications: Survival analysis, reliability engineering.

- **Gamma Distribution**: Generalization of the exponential; models waiting times for multiple events.  
  - Formula:  
    $$
    f(x) = \frac{\beta^\alpha x^{\alpha - 1} e^{-\beta x}}{\Gamma(\alpha)}, \quad x > 0
    $$
  - Used in: Bayesian inference, queuing models.

- **Beta Distribution**: Models probabilities and proportions on $[0,1]$.  
  - Formula:  
    $$
    f(x) = \frac{x^{\alpha - 1} (1-x)^{\beta - 1}}{B(\alpha,\beta)}, \quad 0 < x < 1
    $$
  - Used in: A/B testing, Bayesian modeling.

- **Chi-Square Distribution**: Special case of the gamma distribution; sum of squared standard normal variables.  
  - Formula:  
    $$
    f(x) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2 - 1} e^{-x/2}, \quad x \ge 0
    $$
  - Used in: Hypothesis testing, goodness-of-fit tests.

- **Student's t-Distribution**: Similar to normal but with heavier tails; accounts for extra uncertainty in small samples.  
  - Formula:  
    $$
    f(t) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi} \ \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}
    $$
  - Used in: t-tests, regression analysis.

- **Log-Normal Distribution**: Distribution of a variable whose logarithm is normally distributed.  
  - Formula:  
    $$
    f(x) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}, \quad x > 0
    $$
  - Used in: Modeling positive skewed data, financial returns.

### 3.4 Central Limit Theorem

**Statement**: As sample size increases (and variance is finite), the sample mean distribution approaches normal distribution **regardless of the population distribution's shape**.

#### Breaking It Down:

**What's the Population?**
- Imagine everyone's height, daily expenses, or candies in jars
- Population might have any shape (not necessarily bell curve)

**Take Samples**:
- Pick small groups from population and calculate averages
- Repeat this process many times

**What Happens to Averages?**
- Plot all averages → they form a normal (bell-shaped) curve
- This happens even if original population wasn't bell-shaped!

#### Real-Life Example: Ice Cream Shop

**Scenario**: Daily customer counts are random (5, 20, 50, or 100 customers)

**What Happens**: 
- Take daily average over 30 days and plot those averages
- They will look like a bell curve
- Even though daily counts are random, averages behave predictably

**ML Significance**: Enables z-tests and confidence intervals regardless of original data distribution

---

## 4. Inferential Statistics

### 4.1 Confidence Intervals (CI)

A **confidence interval** represents a range of plausible values for a population parameter based on sample data.  
A **95% confidence interval** means that, in repeated sampling, 95% of the computed intervals would contain the true population parameter.

**Formula for sample mean (unknown $\sigma$)**:  
$$
\bar{x} \pm t_{\alpha/2, \, df=n-1} \cdot \frac{s}{\sqrt{n}}
$$

**Formula for sample mean (known $\sigma$)**:  
$$
\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

Where:  
- $\bar{x}$ = sample mean  
- $s$ = sample standard deviation  
- $\sigma$ = population standard deviation  
- $n$ = sample size  
- $t_{\alpha/2, \, df=n-1}$ = critical value from Student's t-distribution  
- $z_{\alpha/2}$ = critical value from standard normal distribution  
- $\alpha$ = significance level (e.g., 0.05 for 95% CI)

**Key Points**:  
- Use **$t$-distribution** when $\sigma$ is unknown (common in practice).  
- Use **$z$-distribution** when $\sigma$ is known or $n$ is large (Central Limit Theorem).

### 4.2 Hypothesis Testing

**Hypothesis testing** is a statistical method to determine whether there's enough evidence in sample data to infer that a condition is true for the entire population.

#### Workflow:

1. **Formulate Hypotheses**:
   - **H₀ (Null Hypothesis)**: No effect or difference (status quo)
   - **H₁ (Alternative Hypothesis)**: There is an effect

2. **Choose Appropriate Test**:

- **Z-test**:  
  - Use when: Population standard deviation ($\sigma$) is known **or** sample size is large ($n \gtrsim 30$).  
  - Types:  
    - One-sample Z-test (compare sample mean to population mean)  
    - Two-sample Z-test (compare two means with known $\sigma$)  
    - Proportion Z-test (compare proportions)  

- **t-test**:  
  - Use when: $\sigma$ unknown, especially with small sample sizes.  
  - Types:  
    - One-sample t-test (compare sample mean to hypothesized mean)  
    - Independent two-sample t-test (compare means of two independent groups)  
    - Paired t-test (compare means from the same group at two different times)  

- **Chi-square test**:  
  - Use for:  
    - **Independence test** (association between categorical variables)  
    - **Goodness-of-fit** (how well observed frequencies match expected frequencies)  

- **F-test / ANOVA**:  
  - Use for:  
    - **F-test**: Compare two variances.  
    - **ANOVA** (Analysis of Variance): Compare means across 3 or more groups.  
      - One-way ANOVA: One independent variable.  
      - Two-way ANOVA: Two independent variables (can test interaction effects).  

3. **Compute Test Statistic & p-value**

4. **Decision Rule**: p ≤ α (typically 0.05) → reject H₀

#### **P-Value Explained**

The **p-value** (probability value) tells you how likely it is to observe your data (or something more extreme) assuming the null hypothesis is true.

**Simple Explanation**: A p-value tells us how likely results happened just by chance.

##### **Coin Flip Example**:
- **Scenario**: Flip coin 10 times, get 9 heads
- **Normal expectation (H₀)**: Fair coin should get ~5 heads
- **Your result**: 9 heads
- **P-value question**: "If coin is fair, what's the chance of getting 9+ heads just by luck?"

**Interpretation**:
- **Small p-value (e.g., 0.02 = 2%)**: "This almost never happens by luck!" → Maybe coin is rigged (reject H₀)
- **Large p-value (e.g., 0.30 = 30%)**: "This could easily happen by luck" → No proof coin is rigged (keep H₀)

##### **Where P-values are Used in ML**:

**Feature Selection** (Identifying Important Variables):
- In regression models, each feature gets a p-value
- Tells you how likely feature is actually useful (not just noise)
- ✅ Small p-value → Feature likely important
- ❌ Large p-value → Feature may not help much

**Evaluating Statistical Significance**:
- When comparing two models (A vs B), p-value checks:
- "Are result differences real, or due to chance?"

##### **P-value Interpretation Table**:

| p-value | Interpretation |
|---------|----------------|
| ≤ 0.05  | **Strong evidence** against H₀ → Reject H₀ |
| > 0.05  | Weak evidence against H₀ → Fail to reject H₀ |

### 4.3 Correlation vs. Causation

**Key Principle**: High correlation does **NOT** imply causation.

**Example**: Ice cream sales vs. drowning incidents
- High correlation exists (both increase in summer)
- But ice cream doesn't cause drowning
- **Confounding variable**: Hot weather causes both

**ML Implications**:
- Predictive models can use correlated features
- But interventions require causal inference frameworks
- Correlation useful for prediction, causation needed for decision-making

---

## 5. Information Theory

### Entropy

**Definition**:  
Entropy measures the **uncertainty**, **randomness**, or **impurity** in a probability distribution.  
In machine learning, it’s often used in decision trees to measure how mixed a dataset is with respect to target classes.

**Formula**:  
$$
H(X) = -\sum_{i=1}^{n} p_i \log_2 p_i
$$

Where:  
- $X$ = random variable with $n$ possible outcomes  
- $p_i$ = probability of outcome $i$  
- The base of the logarithm determines the units:
  - $\log_2$: Entropy in **bits**
  - $\log_e$: Entropy in **nats**
  - $\log_{10}$: Entropy in **hartleys**

**Properties**:  
- $H(X) \ge 0$  
- $H(X) = 0$ if the outcome is certain (one $p_i = 1$).  
- Maximum entropy occurs when all outcomes are equally likely:  
  $H_{\max} = \log_2(n)$ bits.

**Example**:  
For a fair coin:  
$$
H(X) = -\left[ \frac{1}{2} \log_2 \frac{1}{2} + \frac{1}{2} \log_2 \frac{1}{2} \right] = 1 \ \text{bit}
$$

**Interpretation**:
- **Entropy = 0**: All data points belong to one class (pure)
- **High entropy**: Data evenly split between classes (very uncertain)

**ML Applications**:
- **Decision Trees**: Split on features that reduce entropy most
- **Information Gain**: Measure of entropy reduction
- **Random Forests**: Use entropy for feature selection
- **Feature Selection**: Choose features that maximize information gain

**Example**:
- Dataset with 50% Class A, 50% Class B → High entropy
- Dataset with 90% Class A, 10% Class B → Lower entropy
- Dataset with 100% Class A → Entropy = 0

---

## 6. Summary of Key ML Connections

### Data Quality
- **Random Sampling**: Prevents biased training data
- **Outlier Handling**: Improves model robustness and accuracy

### Feature Engineering  
- **Skewness & Kurtosis**: Guide transformations (log, Box-Cox) to stabilize variance
- **Normality Tests**: Determine appropriate preprocessing for linear models

### Model Evaluation
- **P-values**: Ensure performance differences are statistically significant
- **Confidence Intervals**: Quantify uncertainty in model metrics
- **Hypothesis Tests**: Validate model comparisons aren't due to noise

### Algorithm Selection
- **Entropy**: Core to decision tree splitting criteria
- **Gaussian Assumptions**: Essential for Linear Discriminant Analysis
- **Central Limit Theorem**: Justifies mini-batch gradient descent convergence
- **Bayes' Theorem**: Foundation for probabilistic classifiers

### Statistical Assumptions
- **Normality**: Required for many parametric tests and linear models  
- **Independence**: Critical for valid statistical inference
- **Homoscedasticity**: Equal variance assumption in regression

This comprehensive guide integrates fundamental statistical concepts with their practical applications in machine learning, providing both theoretical understanding and practical implementation guidance.
