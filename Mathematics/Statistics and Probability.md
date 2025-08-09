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
  - Formula: \( \bar{x} = \frac{\sum_{i=1}^{n} x_i}{n} \)
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
- **Variance**: Average squared deviation from mean
  - Formula: \( \sigma^2 = \frac{\sum_{i=1}^{n}(x_i - \mu)^2}{n} \)
- **Standard Deviation**: Square root of variance
  - Formula: \( \sigma = \sqrt{\sigma^2} \)
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

**Karl Pearson's Coefficient of Skewness**:
\[ Sk = \frac{\text{Mean} - \text{Mode}}{\sigma} \]

**Interpretation**:
- Sk > 0: Positively skewed (right tail longer)
- Sk < 0: Negatively skewed (left tail longer)  
- Sk = 0: Symmetric distribution

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
\[ z = \frac{x - \mu}{\sigma} \]
- Outliers typically have |z| > 3

#### 2. **Using Interquartile Range (IQR) Method**
\[ IQR = Q_3 - Q_1 \]
- Outliers: values < Q₁ - 1.5×IQR or > Q₃ + 1.5×IQR

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

- **Sample Space (Ω)**: Set of all possible outcomes
- **Event (A)**: Subset of sample space  
- **Probability**: \( P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}} \)

**Key Probability Rules**:
- **Addition Rule**: \( P(A \cup B) = P(A) + P(B) - P(A \cap B) \)
- **Multiplication Rule**: \( P(A \cap B) = P(A) \cdot P(B|A) \)
- **Conditional Probability**: \( P(A|B) = \frac{P(A \cap B)}{P(B)} \)

### 3.2 Bayes' Theorem

\[ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \]

**Applications in ML**:
- Naive Bayes classifiers
- Bayesian optimization
- Medical diagnosis systems
- Spam filtering

### 3.3 Common Probability Distributions

#### **Discrete Distributions**

- **Bernoulli Distribution**: Single trial with binary outcome
  - Parameter: p (probability of success)
  - Used in: Binary classification

- **Binomial Distribution**: Multiple independent Bernoulli trials
  - Formula: \( P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} \)
  - Applications: A/B testing, quality control

- **Poisson Distribution**: Events occurring in fixed intervals
  - Formula: \( P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \)
  - Used in: Recommendation systems, fraud detection

#### **Continuous Distributions**

- **Normal Distribution**: Bell-shaped curve
  - Formula: \( f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \)
  - Central to many ML algorithms and Central Limit Theorem

- **Standard Normal Distribution**: Special case when μ = 0, σ = 1
  - **Z-score transformation**: \( z = \frac{x - \mu}{\sigma} \)
  - Represents how many standard deviations x is from the mean

- **Uniform Distribution**: All outcomes equally likely
  - Used in: Random sampling, parameter initialization

- **Exponential Distribution**: Time between events
  - Applications: Survival analysis, reliability engineering

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

A **95% confidence interval** around a metric (e.g., accuracy) represents the range that would contain the true value in 95% of repeated samples.

**Formula**: \( \bar{x} \pm t_{\alpha/2} \cdot \frac{s}{\sqrt{n}} \)

### 4.2 Hypothesis Testing

**Hypothesis testing** is a statistical method to determine whether there's enough evidence in sample data to infer that a condition is true for the entire population.

#### Workflow:

1. **Formulate Hypotheses**:
   - **H₀ (Null Hypothesis)**: No effect or difference (status quo)
   - **H₁ (Alternative Hypothesis)**: There is an effect

2. **Choose Appropriate Test**:
   - **Z-test**: Known σ, large sample size
   - **t-test**: Unknown σ or small sample size  
   - **Chi-square test**: Categorical data independence or goodness-of-fit
   - **F-test/ANOVA**: Compare ≥2 variances or means

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

**Definition**: Measures uncertainty or impurity in a dataset.

**Formula**: \[ H(X) = -\sum_{i} p_i \log_2 p_i \]

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
