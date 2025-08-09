# Deep Learning Theory Interview Guide

## Table of Contents
1. [ML Fundamentals](#ml-fundamentals)
2. [Neural Network Fundamentals](#neural-network-fundamentals)
3. [Backpropagation](#backpropagation)
4. [Activation Functions](#activation-functions)
5. [Architectures: CNN, RNN, LSTM, Transformers](#architectures)
6. [Optimization Algorithms](#optimization-algorithms)
7. [Regularization Techniques](#regularization-techniques)
8. [Loss Functions](#loss-functions)
9. [Feature Engineering](#feature-engineering)
10. [Model Evaluation](#model-evaluation)
11. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
12. [Large Language Models](#large-language-models)
13. [Advanced Topics](#advanced-topics)

---

## Large Language Models

### Transformer Architectures

#### BERT (Bidirectional Encoder Representations from Transformers)
- **Architecture**: Encoder-only transformer
- **Training**: Masked Language Modeling (MLM) + Next Sentence Prediction
- **Key feature**: Bidirectional context understanding
- **Use cases**: Text classification, question answering, named entity recognition

#### GPT (Generative Pre-trained Transformer)
- **Architecture**: Decoder-only transformer
- **Training**: Autoregressive next-token prediction
- **Key feature**: Unidirectional (left-to-right) generation
- **Use cases**: Text generation, completion, few-shot learning

### Attention Mechanisms Deep Dive

#### Scaled Dot-Product Attention
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```
- **Scaling factor**: `√d_k` prevents vanishing gradients in softmax

#### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### Long-Range Dependency Solutions
1. **Sparse Attention**: Focus on subset of tokens
2. **Sliding Window Attention**: Fixed-size local windows
3. **Hierarchical Attention**: Multi-level processing

### Positional Encoding

#### RoPE (Rotary Position Embedding)
- **Advantage**: Relative positional encoding
- **Formula**: Rotates query and key vectors based on position
- **Benefits**: Better length generalization

### Efficient Training Techniques

#### LoRA (Low-Rank Adaptation)
![LoRA Architecture](https://miro.medium.com/v2/resize:fit:1400/1*YOTNf-BP-hkJbAo2eGnW0g.png)

- **Concept**: Decompose weight updates into low-rank matrices
- **Formula**: `W = W₀ + BA` where B ∈ ℝⁿˣʳ, A ∈ ℝʳˣᵐ
- **Benefits**: 
  - Dramatically reduces trainable parameters
  - Maintains model performance
  - r is typically 8-64 (much smaller than original dimensions)

### Layer Normalization in Transformers
- **Purpose**: Stabilizes training by normalizing layer inputs
- **Benefits**:
  - Prevents internal covariate shift
  - Enables faster convergence
  - Allows higher learning rates
  - Better gradient flow

### Evaluation Metrics

#### CodeBLEU
- **Purpose**: Evaluate code synthesis quality
- **Components**:
  - N-gram matching (from BLEU)
  - Abstract Syntax Tree (AST) matching
  - Data-flow analysis
- **Advantage**: Better correlation with programmer assessments than BLEU

### LLM vs Traditional Language Models

| Aspect | Traditional LM | Large Language Models |
|--------|----------------|----------------------|
| **Architecture** | N-grams, RNNs | Transformers |
| **Scale** | Millions of parameters | Billions/trillions of parameters |
| **Training** | Task-specific | Pre-training + fine-tuning |
| **Context** | Limited window | Long-range dependencies |
| **Capabilities** | Single task | Multi-task, few-shot learning |

---

## Feature Engineering

![Feature Engineering Process](https://miro.medium.com/v2/resize:fit:1400/1*Hm_5iKKrLlLfIW-dDZAGZA.png)

### Numeric Features

#### Normalization
- **Min-Max Scaling**: Scale to [0,1] or [-1,1]
  ```
  x_norm = (x - x_min)/(x_max - x_min)
  ```

#### Standardization
- **Z-score normalization**: `x_std = (x - μ)/σ`
- **Log transformation**: For power law distributions
  ```
  x_log = log(1 + x/(1 + x_median))
  ```

### Categorical Features

#### Feature Hashing
- **Purpose**: Convert high-cardinality categorical data to fixed-size vectors
- **Benefits**: 
  - Handles categories with 100-1000+ unique values
  - Memory efficient
  - Fast computation
- **Drawbacks**: Hash collisions can hurt model performance

#### Crossed Features
- **Definition**: Combination of two categorical features
- **Example**: `[latitude, longitude]` → `latitude_longitude`
- **Use cases**: Uber location features, Airbnb search ranking

#### Embeddings
- **Purpose**: Dense vector representations of categorical features
- **Benefits**: Capture semantic relationships
- **Dimensionality**: TensorFlow recommends `d = 4√D` where D = number of categories
- **Examples**:
  - Word2Vec for text
  - User embeddings for recommendations
  - Store2Vec at DoorDash

### Handling Missing Data
```python
# Detection
df.isnull().sum()

# Strategies
df.dropna()  # Remove rows
df.fillna(mean_value)  # Imputation
```

### Text Preprocessing

#### Tokenization Strategies
1. **Byte-Pair Encoding (BPE)**: Iteratively merge frequent character pairs
2. **Word-level**: Split on whitespace/punctuation
3. **Character-level**: Individual characters as tokens
4. **Subword**: WordPiece, SentencePiece

---

## Model Evaluation

### Confusion Matrix

![Confusion Matrix](https://miro.medium.com/v2/resize:fit:1400/1*fxiTNIgOyvAombPJx5KGeA.png)

#### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | `(TP + TN)/(TP + TN + FP + FN)` | Overall performance |
| **Precision** | `TP/(TP + FP)` | How accurate positive predictions are |
| **Recall (Sensitivity)** | `TP/(TP + FN)` | Coverage of actual positive samples |
| **Specificity** | `TN/(TN + FP)` | Coverage of actual negative samples |
| **F1 Score** | `2TP/(2TP + FP + FN)` | Harmonic mean of precision/recall |

#### Error Types
- **Type I Error (False Positive)**: Incorrectly classify negative as positive
- **Type II Error (False Negative)**: Incorrectly classify positive as negative

### Model Selection by Dataset Size

#### Small Training Set
- Use models with **high bias, low variance**
- Examples: Naive Bayes, Linear Regression
- Less likely to overfit

#### Large Training Set
- Use models with **low bias, high variance**
- Examples: Neural Networks, Random Forest
- Can capture complex relationships

### Cross-Validation
- **K-Fold**: Split data into k folds, train on k-1, validate on 1
- **Stratified**: Maintains class distribution in each fold
- **Leave-One-Out**: Special case where k = n

---

## Bias-Variance Tradeoff

![Bias-Variance Tradeoff](https://miro.medium.com/v2/resize:fit:1400/1*Y-yJiR0FzMgchPA3hMhDSQ.png)

### Definitions

#### Bias
- **Definition**: Error due to oversimplified assumptions
- **High Bias**: Underfitting, misses relevant patterns
- **Example**: Linear model for non-linear data

#### Variance
- **Definition**: Error due to sensitivity to training data fluctuations
- **High Variance**: Overfitting, models noise
- **Example**: Deep neural network on small dataset

### Total Error Decomposition
```
Total Error = Bias² + Variance + Irreducible Error
```

### Managing the Tradeoff
- **Reduce Bias**: More complex models, more features
- **Reduce Variance**: Regularization, more training data, ensemble methods

---

## ML Fundamentals

### Learning Types

#### Supervised Learning
- **Definition**: Learning with labeled data
- **Examples**: Classification, regression
- **Common algorithms**: Logistic regression, decision trees, SVM, random forest

#### Unsupervised Learning
- **Definition**: Learning patterns from unlabeled data
- **Types**:
  - **Clustering**: K-means, hierarchical clustering
  - **Association**: Market basket analysis, recommendation systems
  - **Dimensionality Reduction**: PCA, t-SNE

#### Semi-supervised Learning
- **Definition**: Small amount of labeled data + large amount of unlabeled data
- **Use cases**: When labeling is expensive (medical imaging, speech recognition)

#### Reinforcement Learning
- **Definition**: Rewards-based learning through interaction with environment
- **Components**: Agent, environment, actions, rewards, policy

### Classification vs Regression
- **Classification**: Predicts categorical outcomes (spam/not spam, cat/dog/bird)
- **Regression**: Predicts continuous values (house prices, temperature, stock prices)

### Classical ML Algorithms

#### K-Nearest Neighbors (kNN)
- **Type**: Supervised classification/regression
- **How it works**: Classifies based on k closest neighbors
- **Pros**: Simple, no assumptions about data
- **Cons**: Computationally expensive, sensitive to irrelevant features

#### Decision Trees
- **Splitting criterion**: Information Gain
- **Formula**: `IG(X;Y) = H(X) - H(X|Y)` where `H(X) = -Σp(x)log₂p(x)`
- **Pruning**: Reduces overfitting by removing branches

#### Random Forest
- **Type**: Ensemble method
- **How it works**: Multiple decision trees + majority voting
- **Advantages**: Reduces overfitting, handles missing values

#### Support Vector Machine (SVM)
- **Objective**: Maximize margin between classes
- **Key concepts**: Support vectors, kernel trick
- **Kernels**: Linear, polynomial, RBF, sigmoid

#### Logistic Regression
- **Formula**: `P(y=1) = 1/(1 + e^(-wᵀx + b))`
- **Use case**: Binary classification
- **Assumptions**: Linear relationship between features and log-odds

#### Linear Regression Assumptions
1. **Multivariate normality**: Residuals are normally distributed
2. **No autocorrelation**: Independence of observations
3. **Homoscedasticity**: Constant variance of residuals
4. **Linear relationship**: Between features and target
5. **No multicollinearity**: Features aren't highly correlated

### Ensemble Learning
- **Definition**: Combining multiple models for better performance
- **Types**:
  - **Bagging**: Random Forest, bootstrap aggregating
  - **Boosting**: AdaBoost, Gradient Boosting, XGBoost
  - **Stacking**: Multiple models with meta-learner

---

## Neural Network Fundamentals

### Basic Structure
A neural network consists of:
- **Input Layer**: Receives raw data
- **Hidden Layers**: Process information through weighted connections
- **Output Layer**: Produces final predictions

![Neural Network Architecture](https://miro.medium.com/v2/resize:fit:1400/1*3fA77_mLNiJTSgZFhYnU0Q.png)

### Mathematical Foundation
For a single neuron:
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
a = f(z)
```
Where:
- `z` = weighted sum (pre-activation)
- `w` = weights
- `x` = inputs
- `b` = bias
- `f` = activation function
- `a` = activation (output)

### Key Concepts
- **Universal Approximation Theorem**: Neural networks with at least one hidden layer can approximate any continuous function
- **Depth vs Width**: Deeper networks can represent more complex functions with fewer parameters
- **Non-linearity**: Essential for learning complex patterns

---

## Backpropagation

### Algorithm Overview
Backpropagation computes gradients of the loss function with respect to each weight using the chain rule.

![Backpropagation Diagram](https://cdn-images-1.medium.com/max/1600/1*q1M7LGiDTirwcS0pG7BzjQ.png)

### Mathematical Formulation
1. **Forward Pass**: Compute activations layer by layer
2. **Backward Pass**: Compute gradients using chain rule

For layer `l`:
```
δˡ = ∂L/∂zˡ = (∂L/∂aˡ) ⊙ f'(zˡ)
∂L/∂Wˡ = δˡ(aˡ⁻¹)ᵀ
∂L/∂bˡ = δˡ
```

### Key Properties
- **Chain Rule**: ∂L/∂w = ∂L/∂z × ∂z/∂w
- **Computational Efficiency**: O(n) time complexity
- **Vanishing Gradients**: Gradients can become exponentially small in deep networks

---

## Activation Functions

### Popular Activation Functions

![Activation Functions Comparison](https://miro.medium.com/v2/resize:fit:1400/1*ZafDv3VUm60Eh10OeJu1vw.png)

#### 1. ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```
**Pros**: Fast computation, helps with vanishing gradient
**Cons**: Dead neurons (always output 0)

#### 2. Leaky ReLU
```
f(x) = max(αx, x) where α is small (e.g., 0.01)
```
**Pros**: Prevents dead neurons
**Cons**: Additional hyperparameter

#### 3. Sigmoid
```
f(x) = 1/(1 + e^(-x))
```
**Pros**: Smooth, probabilistic interpretation
**Cons**: Vanishing gradients, not zero-centered

#### 4. Tanh
```
f(x) = (e^x - e^(-x))/(e^x + e^(-x))
```
**Pros**: Zero-centered, bounded
**Cons**: Still suffers from vanishing gradients

#### 5. Softmax (for multi-class output)
```
f(xᵢ) = e^(xᵢ)/Σⱼe^(xⱼ)
```

### When to Use Which?
- **Hidden layers**: ReLU or variants
- **Binary classification output**: Sigmoid
- **Multi-class classification output**: Softmax
- **Regression output**: Linear (no activation)

---

## Architectures

### Convolutional Neural Networks (CNNs)

![CNN Architecture](https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJC-SA.png)

#### Key Components
1. **Convolution Layer**
   - Applies filters to detect local features
   - Parameters: filter size, stride, padding
   - Feature maps = (Input - Filter + 2×Padding)/Stride + 1

2. **Pooling Layer**
   - Reduces spatial dimensions
   - Max pooling, average pooling
   - Translation invariance

3. **Fully Connected Layer**
   - Traditional neural network layer
   - Usually at the end for classification

#### Popular CNN Architectures
- **LeNet**: Early CNN for digit recognition
- **AlexNet**: First deep CNN to win ImageNet
- **VGG**: Very deep networks with small filters
- **ResNet**: Skip connections to enable very deep networks

### Recurrent Neural Networks (RNNs)

![RNN Architecture](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png)

#### Basic RNN
```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

**Problems**:
- **Vanishing Gradients**: Gradients diminish exponentially
- **Exploding Gradients**: Gradients grow exponentially

### Long Short-Term Memory (LSTM)

![LSTM Cell](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

#### LSTM Gates
1. **Forget Gate**: What information to discard
   ```
   f_t = σ(W_f × [h_{t-1}, x_t] + b_f)
   ```

2. **Input Gate**: What new information to store
   ```
   i_t = σ(W_i × [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)
   ```

3. **Output Gate**: What parts of cell state to output
   ```
   o_t = σ(W_o × [h_{t-1}, x_t] + b_o)
   ```

#### Cell State Update
```
C_t = f_t * C_{t-1} + i_t * C̃_t
h_t = o_t * tanh(C_t)
```

### Transformers

![Transformer Architecture](https://jalammar.github.io/images/t/the_transformer_3.png)

#### Key Innovation: Self-Attention
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

Where:
- Q = Queries, K = Keys, V = Values
- All derived from input through learned transformations

#### Multi-Head Attention
- Run multiple attention heads in parallel
- Each head learns different types of relationships
- Concatenate and project results

#### Advantages over RNNs
- **Parallelization**: Can process entire sequence simultaneously
- **Long-range dependencies**: Direct connections between all positions
- **Scalability**: Scales well with increased compute

---

## Optimization Algorithms

### Gradient Descent Variants

![Optimization Algorithms Comparison](https://ruder.io/content/images/2016/09/contours_evaluation_optimizers.gif)

#### 1. Stochastic Gradient Descent (SGD)
```
w = w - η∇w
```
- Simple but can be slow
- High variance in updates

#### 2. SGD with Momentum
```
v_t = γv_{t-1} + η∇w
w = w - v_t
```
- Accelerates in consistent directions
- Dampens oscillations

#### 3. AdaGrad
```
G_t = G_{t-1} + (∇w)²
w = w - η/(√G_t + ε) × ∇w
```
- Adapts learning rate per parameter
- Good for sparse data

#### 4. RMSprop
```
E[g²]_t = γE[g²]_{t-1} + (1-γ)(∇w)²
w = w - η/(√E[g²]_t + ε) × ∇w
```
- Fixes AdaGrad's diminishing learning rates
- Uses exponential moving average

#### 5. Adam (Adaptive Moment Estimation)
```
m_t = β₁m_{t-1} + (1-β₁)∇w     # First moment
v_t = β₂v_{t-1} + (1-β₂)(∇w)²  # Second moment
m̂_t = m_t/(1-β₁ᵗ)              # Bias correction
v̂_t = v_t/(1-β₂ᵗ)              # Bias correction
w = w - η × m̂_t/(√v̂_t + ε)
```

**Default hyperparameters**: β₁=0.9, β₂=0.999, ε=1e-8

---

## Regularization Techniques

### 1. Dropout

![Dropout Visualization](https://miro.medium.com/v2/resize:fit:1400/1*iWQzxhVlvadk6VAJjsgXgg.png)

#### How it works
- Randomly set neurons to 0 during training
- Scale remaining neurons by 1/p (where p = keep probability)
- Forces network to not rely on specific neurons

#### Implementation
```python
# Training
if training:
    mask = np.random.binomial(1, keep_prob, size=hidden.shape)
    hidden = hidden * mask / keep_prob
```

### 2. Batch Normalization

![Batch Normalization](https://miro.medium.com/v2/resize:fit:1400/1*Hiq-rLFGDpESpr8QNsJ1jg.png)

#### Algorithm
For each mini-batch:
```
μ = (1/m)Σxᵢ                    # Mean
σ² = (1/m)Σ(xᵢ - μ)²            # Variance
x̂ᵢ = (xᵢ - μ)/√(σ² + ε)        # Normalize
yᵢ = γx̂ᵢ + β                   # Scale and shift
```

#### Benefits
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization
- Reduces dependence on initialization

#### Lasso vs Ridge Regression
- **Lasso (L1)**: `Loss + λΣ|w_i|`
  - Encourages sparsity (feature selection)
  - Can drive coefficients to exactly zero
- **Ridge (L2)**: `Loss + λΣw_i²`
  - Shrinks coefficients toward zero
  - Handles multicollinearity better

### 3. L1 and L2 Regularization

#### L2 Regularization (Weight Decay)
```
Loss = Original_Loss + λ/2 × Σwᵢ²
```
- Penalizes large weights
- Encourages weight diffusion

#### L1 Regularization
```
Loss = Original_Loss + λ × Σ|wᵢ|
```
- Encourages sparsity
- Feature selection

### 4. Early Stopping
- Monitor validation loss during training
- Stop when validation loss starts increasing
- Prevents overfitting without explicit regularization

### 5. Data Augmentation
- Artificially increase training data
- Image: rotation, scaling, flipping
- Text: synonym replacement, back-translation

---

## Loss Functions

### Classification Losses

#### 1. Cross-Entropy Loss
**Binary**: 
```
L = -[y log(p) + (1-y) log(1-p)]
```

**Multi-class**:
```
L = -Σᵢ yᵢ log(pᵢ)
```

#### 2. Hinge Loss (SVM)
```
L = max(0, 1 - y × f(x))
```

### Regression Losses

#### 1. Mean Squared Error (MSE)
```
L = (1/n)Σ(yᵢ - ŷᵢ)²
```

#### 2. Mean Absolute Error (MAE)
```
L = (1/n)Σ|yᵢ - ŷᵢ|
```

#### 3. Huber Loss
```
L = {
    0.5(y - ŷ)²           if |y - ŷ| ≤ δ
    δ|y - ŷ| - 0.5δ²      otherwise
}
```

---

## Advanced Topics

### 1. Attention Mechanisms

#### Scaled Dot-Product Attention
```
Attention(Q,K,V) = softmax(QK^T/√dₖ)V
```

#### Self-Attention
- Q, K, V all come from the same input
- Allows modeling dependencies regardless of distance

### 2. Residual Connections (Skip Connections)

![ResNet Block](https://miro.medium.com/v2/resize:fit:786/1*D0F3UitQ2l5Q0Ak-tjEdJg.png)

```
F(x) = H(x) - x
H(x) = F(x) + x
```

**Benefits**:
- Solves vanishing gradient problem
- Enables training of very deep networks
- Identity mapping preserves information

### 3. Normalization Techniques

#### Layer Normalization
- Normalizes across features for each sample
- Used in transformers and RNNs

#### Group Normalization
- Divides channels into groups and normalizes within groups
- Less dependent on batch size

### 4. Overfitting vs Underfitting

#### Overfitting
- **Symptoms**: High training accuracy, low validation accuracy
- **Causes**: Model too complex, insufficient data, too many epochs
- **Solutions**: 
  - Regularization (L1/L2, dropout)
  - Cross-validation
  - Early stopping
  - More training data

#### Underfitting
- **Symptoms**: Low training and validation accuracy
- **Causes**: Model too simple, insufficient features
- **Solutions**:
  - Increase model complexity
  - Add more features
  - Reduce regularization

### 5. Boltzmann Machines
- **Structure**: Fully connected neural network
- **Layers**: Visible and hidden units
- **Learning**: Energy-based model using Gibbs sampling
- **Applications**: Dimensionality reduction, collaborative filtering

### 6. Principal Component Analysis (PCA)
- **Purpose**: Dimensionality reduction
- **Method**: Project data onto principal components (eigenvectors)
- **Benefits**: Removes correlation, reduces noise
- **Limitation**: Linear transformation only

---

## Advanced Topics

### 1. Three Stages of ML Model Development

#### Stage 1: Model Building
- Choose appropriate algorithm
- Feature engineering and selection
- Hyperparameter tuning
- Training on training set

#### Stage 2: Model Testing
- Evaluate on validation/test set
- Check for overfitting/underfitting
- Compare different models
- Cross-validation

#### Stage 3: Model Deployment
- Make final adjustments
- Deploy to production
- Monitor performance
- Retrain as needed

### 2. Correlation vs Covariance

#### Covariance
```
Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]
```
- Measures direction of linear relationship
- Units depend on original variables

#### Correlation
```
Corr(X,Y) = Cov(X,Y)/(σₓσᵧ)
```
- Normalized covariance [-1, 1]
- Measures both strength and direction

### 3. Gradient Problems Solutions

#### Vanishing Gradients
- **Problem**: Gradients become exponentially small
- **Solutions**: 
  - ReLU activations
  - Residual connections
  - LSTM/GRU for RNNs
  - Proper weight initialization

#### Exploding Gradients
- **Problem**: Gradients become exponentially large
- **Solutions**:
  - Gradient clipping
  - Lower learning rates
  - Weight regularization

### 4. Data Splitting Strategies
- **Standard**: 70/15/15 or 80/10/10 (train/val/test)
- **Time series**: Chronological split
- **Stratified**: Maintains class distribution
- **Cross-validation**: Multiple train/val splits

---

## Real-World Applications

### Email Spam Detection
1. **Data Collection**: Thousands of labeled emails
2. **Feature Extraction**: Word frequency, metadata, headers
3. **Model Training**: Decision trees, SVM, Naive Bayes
4. **Deployment**: Real-time classification with probability threshold

### Healthcare Diagnosis
1. **Data**: Medical images, patient records
2. **Preprocessing**: Image normalization, data augmentation
3. **Models**: CNNs for imaging, ensemble methods
4. **Validation**: Cross-validation with medical expert review

### Sentiment Analysis
1. **Data**: Text with sentiment labels (positive/negative/neutral)
2. **Preprocessing**: Tokenization, stemming, stop word removal
3. **Models**: LSTM, BERT, transformer-based
4. **Applications**: Social media monitoring, product reviews

---

## Interview Tips

### Common Questions
1. **Explain backpropagation in detail**
2. **Why do we use non-linear activation functions?**
3. **What's the difference between batch norm and layer norm?**
4. **How does attention work in transformers?**
5. **Explain the vanishing gradient problem**
6. **What's the bias-variance tradeoff?**
7. **How do you handle overfitting?**
8. **Explain the difference between L1 and L2 regularization**
9. **How does BERT differ from GPT?**
10. **What is the purpose of feature engineering?**

### ML vs Deep Learning Comparison

| Aspect | Machine Learning | Deep Learning |
|--------|------------------|---------------|
| **Feature Engineering** | Manual | Automatic |
| **Data Requirements** | Small to medium | Large |
| **Computational Power** | Low to medium | High |
| **Interpretability** | Higher | Lower |
| **Problem Solving** | Divide and conquer | End-to-end |
| **Examples** | SVM, Random Forest | Neural Networks |

### Key Points to Remember
- **Always explain the intuition** before diving into math
- **Discuss trade-offs** of different approaches
- **Mention recent developments** (e.g., attention mechanisms)
- **Connect theory to practice** with examples

### Mathematical Notation
- Familiarize yourself with standard notation
- Be comfortable with matrix operations
- Understand chain rule applications

---

## Quick Reference Formulas

### Backpropagation
```
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
```

### Adam Optimizer
```
m_t = β₁m_{t-1} + (1-β₁)∇w
v_t = β₂v_{t-1} + (1-β₂)(∇w)²
w_t = w_{t-1} - η × m̂_t/(√v̂_t + ε)
```

### Batch Normalization
```
BN(x) = γ × (x - μ)/σ + β
```

### Cross-Entropy Loss
```
L = -Σ y_true × log(y_pred)
```

This guide covers the essential deep learning concepts you'll need for ML theory interviews. Focus on understanding the intuition behind each concept and be prepared to explain trade-offs and implementation details.