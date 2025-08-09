# Search Engines & ML Theory Interview Prep Guide

## Table of Contents

### Core Information Retrieval Concepts
- [TF-IDF (Term Frequency-Inverse Document Frequency)](#tf-idf-term-frequency-inverse-document-frequency)
- [BM25 (Best Matching 25)](#bm25-best-matching-25)
- [Structured Queries](#structured-queries)

### Retrieval Models & Ranking
- [Language Models (Indri)](#language-models-indri)
  - [Dirichlet Smoothing](#dirichlet-smoothing)
  - [Jelinek-Mercer Smoothing](#jelinek-mercer-smoothing)
  - [Why Smoothing Matters](#why-smoothing-matters)
- [Sequential Dependency Models (SDM)](#sequential-dependency-models-sdm)

### Learning to Rank (LTR)
- [Approaches](#approaches)
  - [Pointwise](#pointwise)
  - [Pairwise](#pairwise)
  - [Listwise](#listwise)
- [Popular LTR Algorithms](#popular-ltr-algorithms)
- [Position and Presentation Bias](#position-and-presentation-bias-in-learning-to-rank)
- [Feature Engineering](#feature-engineering)
- [Evaluation Metrics](#evaluation-metrics)
  - [P@k (Precision at k)](#pk-precision-at-k)
  - [MAP (Mean Average Precision)](#map-mean-average-precision)
  - [MRR (Mean Reciprocal Rank)](#mrr-mean-reciprocal-rank)
  - [NDCG (Normalized Discounted Cumulative Gain)](#ndcg-normalized-discounted-cumulative-gain)

### Recommendation Systems

![Recommendation Systems Overview](https://miro.medium.com/v2/resize:fit:1400/1*Zvwzw0FJaG_OZ8bGAXhfGQ.png)
- [Collaborative Filtering](#collaborative-filtering)
- [Content-Based Filtering](#content-based-filtering)
- [Embedding-Based Approaches](#embedding-based-approaches)
- [Two-Tower Architecture](#two-tower-architecture)
- [Personalization Algorithms](#personalization-algorithms)
- [Production Considerations](#production-considerations)

### Diversity & Personalization

![Diversity in Search Results](https://research.google/static/images/publications/intent-aware-query-obfuscation-for-privacy-protection-in-personalized-web-search_v1_01.gif)
- [Diversity Metrics](#diversity-metrics)
  - [α-NDCG](#α-ndcg-alpha-normalized-discounted-cumulative-gain)
  - [P-IA@k](#p-iak-intent-aware-precision-at-k)
  - [Subtopic Coverage](#subtopic-coverage)
- [Diversification Algorithms](#diversification-algorithms)
  - [PM2 (Portfolio Model 2)](#pm2-portfolio-model-2)
  - [xQuAD](#xquad-explicit-query-aspect-diversification)
  - [MMR (Maximal Marginal Relevance)](#mmr-maximal-marginal-relevance)

### Advanced Topics
- [Neural Information Retrieval](#neural-information-retrieval)
- [Modern Production Systems](#modern-production-systems)
- [LLM Integration](#llm-integration-in-searchrecsys)
- [A/B Testing for Recommender Systems](#ab-testing-for-recommender-systems)

### System Design Examples

![System Design Overview](https://miro.medium.com/v2/resize:fit:1400/1*ZkwKKz0p2Cl8BgBTUC4ebA.png)
- [Facebook News Feed](#facebook-news-feed)
- [YouTube (Long-form Video)](#youtube-long-form-video)
- [TikTok (Short-form Video)](#tiktok-short-form-video)
- [Google Search Engine](#google-search-engine)

### Common Interview Questions
- [Algorithm Questions & Solutions](#common-algorithm-questions--solutions)
- [Technical Deep Dives](#technical-deep-dives)
- [System Design Case Study: TikTok](#system-design-case-study-tiktok-recommendation-engine)

### Key Takeaways
- [Interview Best Practices](#key-takeaways-for-interviews)

---

## Core Information Retrieval Concepts

### TF-IDF (Term Frequency-Inverse Document Frequency)
- **Definition**: Measures word importance by combining term frequency in document with rarity across collection
- **Formula**: 
  - **Simple**: `TF-IDF(t,d) = TF(t,d) × IDF(t) = (count(t,d) / |d|) × log(N / df(t))`
  - **LaTeX**: $\text{TF-IDF}(t,d) = \frac{\text{count}(t,d)}{|d|} \times \log\left(\frac{N}{\text{df}(t)}\right)$

![TF-IDF Concept](https://miro.medium.com/v2/resize:fit:1400/1*qQgnyPLDIkUmeZKN2_ZWbQ.png)

- **Use Cases**: Document ranking, feature extraction for ML models
- **Limitations**: Doesn't handle synonyms, ignores word order
- **Interview Q**: "How would you modify TF-IDF for very short queries vs long documents?"
- **Practice**: Implement TF-IDF scoring function
  ```python
  def tf_idf(term, doc, corpus):
      tf = doc.count(term) / len(doc)
      df = sum(1 for d in corpus if term in d)
      idf = math.log(len(corpus) / df)
      return tf * idf
  ```

### BM25 (Best Matching 25)
- **Definition**: Probabilistic ranking function that improves on TF-IDF
- **Formula**: 
  - **Simple**: `BM25(q,d) = Σ IDF(qi) × (f(qi,d) × (k1+1)) / (f(qi,d) + k1 × (1-b + b × |d|/avgdl))`
  - **LaTeX**: $\text{BM25}(q,d) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i,d) \cdot (k_1 + 1)}{f(q_i,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$

![BM25 vs TF-IDF Comparison](https://static.wixstatic.com/media/3eee0b_f1c1b0f905bd4359b54e97bf13e3bf67~mv2.png)

- **Parameters**: 
  - k1 (1.2-2.0): Controls term frequency saturation
  - b (0.75): Controls length normalization
- **Advantages**: Document length normalization, saturation effects
- **Interview Q**: "Why does BM25 perform better than TF-IDF for varying document lengths?"
- **Practice**: Implement BM25 scoring function
  ```python
  def bm25_score(query_terms, doc, corpus, k1=1.2, b=0.75):
      doc_len = len(doc)
      avg_doc_len = sum(len(d) for d in corpus) / len(corpus)
      score = 0
      for term in query_terms:
          tf = doc.count(term)
          df = sum(1 for d in corpus if term in d)
          idf = math.log((len(corpus) - df + 0.5) / (df + 0.5))
          norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
          score += idf * norm
      return score
  ```

### Structured Queries
- **Boolean Operators**: AND, OR, NOT
- **Proximity Operators**: NEAR/n, WINDOW/n
- **Field-Specific**: title:, url:, keywords:
- **Synonyms**: #syn(term1 term2 term3)
- **Example**: `#and(#near/10(machine learning) #syn(AI artificial intelligence))`
- **Interview Q**: "Design a structured query for finding recent papers about neural networks in computer vision"

## Retrieval Models & Ranking

### Language Models (Indri)
#### Dirichlet Smoothing
- **Formula**: 
  - **Simple**: `P(t|d) = (c(t,d) + μ × P(t|C)) / (|d| + μ)`
  - **LaTeX**: $P(t|d) = \frac{c(t,d) + \mu \cdot P(t|C)}{|d| + \mu}$
- **What it does**: Adds a constant amount of smoothing based on collection probability
- **Effect**: Longer documents need less smoothing (more reliable statistics)
- **Good for**: Variable length documents, μ=1500 is typical

#### Jelinek-Mercer Smoothing
- **Formula**: 
  - **Simple**: `P(t|d) = λ × P(t|d_ml) + (1-λ) × P(t|C)`
  - **LaTeX**: $P(t|d) = \lambda \cdot P(t|d_{ml}) + (1-\lambda) \cdot P(t|C)$
- **What it does**: Linear interpolation between document and collection models
- **Effect**: Fixed proportion of smoothing regardless of document length
- **Good for**: When you want consistent smoothing, λ=0.4 is typical

#### Why Smoothing Matters
- **Zero probability problem**: Unseen terms get zero probability
- **Data sparsity**: Short documents have unreliable statistics
- **Ranking implications**: Without smoothing, documents missing query terms get zero score

### Sequential Dependency Models (SDM)
- **Components**: Unigrams, Bigrams, Windows
- **Weights**: Typically λ1=0.8, λ2=0.1, λ3=0.1 for unigram/bigram/window
- **Formula**: 
  - **Simple**: `P(Q|D) = λ1×P(q1,q2,...|D) + λ2×P(#near(q1,q2)|D) + λ3×P(#window(q1,q2)|D)`
  - **LaTeX**: $P(Q|D) = \lambda_1 \sum_{i} P(q_i|D) + \lambda_2 \sum_{i} P(\#\text{near}(q_i,q_{i+1})|D) + \lambda_3 \sum_{i} P(\#\text{window}(q_i,q_{i+1})|D)$
- **Interview Q**: "How would you tune SDM parameters for different query types?"

## Learning to Rank (LTR)

![Learning to Rank Overview](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Features/images/ranking.png)

### Approaches

#### Pointwise
- **Concept**: Predict relevance score for each document independently
- **Examples**: Regression, classification on individual docs
- **Pros**:
  - Simple to implement and understand
  - Can leverage standard ML algorithms
  - Good when you have absolute relevance labels
- **Cons**:
  - Ignores relative ordering between documents
  - Doesn't directly optimize ranking metrics
  - May not handle ranking-specific issues well

#### Pairwise
- **Concept**: Learn relative preferences between document pairs
- **Examples**: RankNet, SVMRank, LambdaRank
- **What it optimizes**: Pairwise ranking accuracy, NOT directly NDCG
  - **RankNet**: Pairwise cross-entropy loss
  - **SVMRank**: Hinge loss on violated pairs
  - **Problem**: Treats all pair violations equally, but NDCG cares more about top-position errors
- **Pros**:
  - More robust to label noise than pointwise
  - Natural for preference data (clicks at different positions)
  - Good for sparse interaction data
  - Easier to understand conceptually
- **Cons**:
  - Doesn't directly optimize ranking metrics like NDCG
  - Can generate many training pairs (computational overhead)
  - Position bias in click data needs careful handling

#### Listwise
- **Concept**: Optimize entire ranking list simultaneously
- **Examples**: ListNet, LambdaMART, AdaRank
- **What it optimizes**: Directly optimizes ranking metrics (NDCG, MAP)
- **Pros**:
  - Directly optimizes the metric you care about
  - Actually handles sparse data well (contrary to intuition)
  - Fewer training examples than pairwise (one per query vs many pairs)
  - Industry standard (LambdaMART widely used in production)
- **Cons**:
  - More complex to implement from scratch
  - Can be harder to debug than pairwise approaches
- **Reality check**: Most production systems use listwise methods despite theoretical concerns about sparsity

### Popular LTR Algorithms
- **SVMRank**: Pairwise approach, optimizes hinge loss on ranking violations
- **RankNet**: Neural network pairwise approach with cross-entropy loss
- **LambdaMART**: Gradient boosting that directly optimizes NDCG (industry standard)
- **Coordinate Ascent**: Iteratively optimizes one feature at a time
- **ListNet**: Uses neural networks to model probability distributions over permutations

### Position and Presentation Bias in Learning to Rank

#### The Problem
User clicks don't always reflect true relevance due to biases:
- **Position bias**: Users more likely to click higher positions regardless of relevance
- **Presentation bias**: How the result is displayed (title, snippet) affects clicks
- **Example**: Highly relevant document at rank 1 with poor title gets fewer clicks than less relevant document at rank 2 with attractive title

#### Solutions in Production
1. **Randomization**: Occasionally show results in random order to get unbiased click data
2. **A/B testing**: Test different snippets/titles for same document
3. **Dwell time weighting**: Clicks + time spent is stronger signal than just clicks
4. **Multiple signals**: Combine clicks with bookmarks, shares, return visits
5. **Inverse Propensity Scoring**: Weight training examples by `1/P(click|position)`
6. **Position-weighted training**: Weight pairwise violations by position distance

#### Practical Implementation
```python
def extract_training_pairs_with_bias_correction(query_sessions):
    pairs = []
    for session in query_sessions:
        for i, (doc_i, clicked_i, dwell_i) in enumerate(session.results):
            for j, (doc_j, clicked_j, dwell_j) in enumerate(session.results):
                if clicked_i and not clicked_j and i > j:  # Lower position clicked
                    # Weight by position distance and dwell time
                    weight = (i - j) * max(dwell_i / avg_dwell_time, 1.0)
                    pairs.append((doc_i, doc_j, weight))
    return pairs
```

### Feature Engineering
- **Query-Document Features**: BM25, TF-IDF, language model scores
- **Query Features**: Length, type, ambiguity
- **Document Features**: PageRank, length, quality signals
- **Custom Features** (from homework):
  - URL complexity (separator count)
  - Keyword field scoring
  - Keyword density in document
- **Practice**: Design features for learning-to-rank system
  ```python
  def extract_features(query, doc, corpus):
      features = []
      # Relevance features
      features.append(bm25_score(query, doc, corpus))
      features.append(tf_idf_score(query, doc, corpus))
      # Document quality features
      features.append(len(doc))  # Document length
      features.append(doc.url_complexity())  # URL separators
      # Query-doc interaction
      features.append(query_doc_overlap(query, doc))
      return features
  ```

### Evaluation Metrics

#### P@k (Precision at k)
**Formula**: 
- **Simple**: `P@k = (number of relevant docs in top-k) / k`
- **LaTeX**: $P@k = \frac{\text{number of relevant docs in top-}k}{k}$
- **Use case**: When you care about precision at a specific cutoff
- **Limitation**: Doesn't consider ranking order within top-k

#### MAP (Mean Average Precision)
**Formula**: 
- **Simple**: `MAP = (1/|Q|) × Σ AP(q)` where `AP(q) = (1/R) × Σ P@k × rel(k)`
- **LaTeX**: $\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}(q)$
  $\text{AP}(q) = \frac{1}{R} \sum_{k=1}^{n} P@k \cdot \text{rel}(k)$
- **Components**: 
  - R = total number of relevant documents for query q
  - rel(k) = 1 if document at rank k is relevant, 0 otherwise
  - P@k = precision at rank k
- **Pros**: 
  - Considers all relevant documents
  - Position-sensitive (earlier relevant docs weighted more)
- **Cons**: 
  - Binary relevance only (relevant vs not relevant)
  - Assumes user wants to find ALL relevant documents
- **Best for**: Information retrieval tasks where recall matters

#### MRR (Mean Reciprocal Rank)
**Formula**: 
- **Simple**: `MRR = (1/|Q|) × Σ (1/rank_i)` where rank_i is position of first relevant document
- **LaTeX**: $\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$
- **Pros**: 
  - Simple and intuitive
  - Good for navigational queries
- **Cons**: 
  - Only cares about first relevant result
  - Ignores other relevant documents
- **Best for**: Navigational queries ("Facebook login", "weather today")

#### NDCG (Normalized Discounted Cumulative Gain)
**Formula**: 
- **Simple**: `NDCG@k = DCG@k / IDCG@k`
- **LaTeX**: $\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$

**DCG**: 
- **Simple**: `DCG@k = Σ(i=1 to k) (2^rel_i - 1) / log2(i + 1)`
- **LaTeX**: $\text{DCG}@k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$

**IDCG**: Ideal DCG (DCG of perfect ranking)

![NDCG Visualization](https://miro.medium.com/v2/resize:fit:1400/1*jQKdtT3ol5b4n9_j8gWwaw.png)

- **Components**:
  - rel_i = relevance grade of document at position i (e.g., 0-4 scale)
  - Logarithmic position discount (position 2 gets ~half weight of position 1)
- **Pros**: 
  - Handles graded relevance (not just binary)
  - Position-sensitive with principled discounting
  - Normalized (0-1 scale, comparable across queries)
- **Cons**: 
  - Requires graded relevance judgments
  - More complex to interpret than precision/recall
- **Best for**: Modern search engines with multiple relevance levels

**Interview Q**: "When would you prefer NDCG over MAP?"
- **NDCG when**: You have graded relevance labels and position matters a lot
- **MAP when**: Binary relevance and you care about finding all relevant docs

## Recommendation Systems

### Collaborative Filtering
- **User-Based**: Find similar users, recommend what they liked
- **Item-Based**: Find similar items to what user liked
- **Matrix Factorization**: SVD, NMF to find latent factors
- **Cold Start Problem**: New users/items with no interaction data
- **Practice**: Implement collaborative filtering with matrix factorization
  ```python
  def matrix_factorization(R, K=10, steps=5000, alpha=0.0002, beta=0.02):
      N, M = R.shape
      P = np.random.normal(scale=1./K, size=(N, K))
      Q = np.random.normal(scale=1./K, size=(M, K))
      
      for step in range(steps):
          for i, j in zip(*R.nonzero()):
              eij = R[i,j] - np.dot(P[i,:], Q[j,:])
              P[i,:] += alpha * (eij * Q[j,:] - beta * P[i,:])
              Q[j,:] += alpha * (eij * P[i,:] - beta * Q[j,:])
      return P, Q
  ```

### Content-Based Filtering
- **Approach**: Recommend items similar to user's previous preferences
- **Features**: Item attributes (genre, director, price, etc.)
- **Advantages**: No cold start for new users, explainable
- **Limitations**: Over-specialization, limited serendipity

### Embedding-Based Approaches
- **Concept**: Learn dense vector representations for users/items
- **Training**: Neural networks learn embeddings from interaction data
- **Similarity**: Cosine similarity or dot product in embedding space
- **Advantages**: Capture complex, non-linear relationships

### Two-Tower Architecture
- **Design**: Separate neural networks for users and items
- **User Tower**: Processes user features → user embedding
- **Item Tower**: Processes item features → item embedding
- **Scoring**: Dot product of embeddings

![Two-Tower Architecture](https://miro.medium.com/v2/resize:fit:1400/1*NNOdVrIKxJ4QNMTjqGWz6Q.png)

- **Advantages**: Scalable inference, pre-computable item embeddings
- **Trade-offs**: Limited cross-feature interactions
- **Practice**: Build a simple two-tower recommendation model
  ```python
  class TwoTowerModel(nn.Module):
      def __init__(self, user_features, item_features, embedding_dim):
          super().__init__()
          self.user_tower = nn.Sequential(
              nn.Linear(user_features, 128),
              nn.ReLU(),
              nn.Linear(128, embedding_dim)
          )
          self.item_tower = nn.Sequential(
              nn.Linear(item_features, 128),
              nn.ReLU(),
              nn.Linear(128, embedding_dim)
          )
      
      def forward(self, user_feat, item_feat):
          user_emb = self.user_tower(user_feat)
          item_emb = self.item_tower(item_feat)
          return torch.sum(user_emb * item_emb, dim=1)
  ```

### Personalization Algorithms

#### Context-Aware Recommendations
- **Time-based**: Different recommendations for different times of day
- **Location-based**: Geographic preferences and constraints
- **Device-based**: Mobile vs desktop behavior patterns
- **Session-based**: Sequential pattern mining within sessions

#### Multi-Armed Bandits
- **ε-greedy**: Explore random recommendations with probability ε
- **UCB (Upper Confidence Bound)**: Balance exploration and exploitation
- **Thompson Sampling**: Bayesian approach to exploration
- **Contextual Bandits**: Incorporate user/item context into bandit algorithms

#### Deep Learning for Personalization
- **Autoencoders**: Learn user/item representations from interactions
- **Neural Collaborative Filtering**: Replace matrix factorization with neural networks
- **Recurrent Networks**: Model sequential user behavior
- **Attention Mechanisms**: Focus on relevant past interactions

#### Real-time Personalization
- **Online Learning**: Update models with each new interaction
- **Incremental Learning**: Efficient updates without full retraining
- **Feature Stores**: Real-time feature computation and serving
- **A/B Testing**: Continuous experimentation with personalization strategies

### Production Considerations
- **Multi-Task Models**: Single model for multiple recommendation scenarios
- **LLM Integration**: Feature generation, content understanding, explanations
- **Serving**: Two-stage (candidate generation + ranking)
- **Evaluation**: A/B testing, online metrics vs offline metrics

## Diversity & Personalization

### Diversity Metrics

#### α-NDCG (Alpha-Normalized Discounted Cumulative Gain)
**Formula**:
- **Simple**: 
```
α-NDCG@k = (1/Z) × Σ(i=1 to k) [(2^rel(di) - 1) / log2(i+1)] × Π(j=1 to i-1) (1 - α × I(dj, di))
```
- **LaTeX**: 
$\alpha\text{-NDCG}@k = \frac{1}{Z} \sum_{i=1}^{k} \frac{2^{\text{rel}(d_i)} - 1}{\log_2(i+1)} \prod_{j=1}^{i-1} (1 - \alpha \cdot I(d_j, d_i))$
Where:
- rel(di) = relevance of document di
- I(dj, di) = indicator function (1 if documents dj and di cover the same subtopic, 0 otherwise)
- α = diversity parameter (0 ≤ α ≤ 1)
- Z = normalization factor (ideal α-NDCG@k)
- When α = 0: reduces to standard NDCG (no diversity penalty)
- When α = 1: maximum diversity penalty for redundant documents

#### P-IA@k (Intent-Aware Precision at k)
**Formula**:
```
P-IA@k = (1/k) × Σ(i=1 to k) Σ(s∈S) P(s|q) × rel(di, s)
```
Where:
- P(s|q) = probability of subtopic s given query q
- rel(di, s) = relevance of document di to subtopic s (binary: 0 or 1)
- S = set of all subtopics for query q
- k = number of documents retrieved

**Alternative formulation**:
```
P-IA@k = Σ(s∈S) P(s|q) × [number of relevant docs for subtopic s in top-k / k]
```

#### Subtopic Coverage
**Formula**:
- **Simple**: 
```
Subtopic Coverage@k = |{s ∈ S : ∃di ∈ Dk, rel(di, s) = 1}| / |S|
```
- **LaTeX**: 
$\text{Subtopic Coverage}@k = \frac{|\{s \in S : \exists d_i \in D_k, \text{rel}(d_i, s) = 1\}|}{|S|}$
Where:
- S = set of all subtopics for the query
- Dk = set of top-k retrieved documents
- rel(di, s) = 1 if document di is relevant to subtopic s
- |·| denotes cardinality (set size)

**Interpretation**: Fraction of query subtopics that have at least one relevant document in the top-k results

### Diversification Algorithms

#### PM2 (Portfolio Model 2)
**Formula**:
- **Simple**: 
```
PM2(d|q) = λ × P(d|q) + (1-λ) × Σ P(s|q) × P(d|s) × (1 - Π (1 - P(di|s)))
```
- **LaTeX**: 
$\text{PM2}(d|q) = \lambda \cdot P(d|q) + (1-\lambda) \sum_{s} P(s|q) \cdot P(d|s) \cdot \left(1 - \prod_{d_i \in S} (1 - P(d_i|s))\right)$
Where:
- P(d|q) = relevance of document d to query q
- P(s|q) = probability of subtopic s given query q
- P(d|s) = probability of document d covering subtopic s
- λ controls relevance vs diversity trade-off

#### xQuAD (eXplicit Query Aspect Diversification)
**Formula**:
- **Simple**: 
```
xQuAD(d|q) = (1-λ) × P(d|q) + λ × Σ P(s|q) × P(d|s) × Π (1 - P(di|s))
```
- **LaTeX**: 
$\text{xQuAD}(d|q) = (1-\lambda) \cdot P(d|q) + \lambda \sum_{s} P(s|q) \cdot P(d|s) \cdot \prod_{d_i \in S} (1 - P(d_i|s))$
Where:
- Similar to PM2 but different normalization
- Π (1 - P(di|s)) represents how much subtopic s is already covered

### PM2 vs xQuAD Comparison

#### Key Differences

**1. Coverage Term Interpretation**:
- **PM2**: `(1 - Π (1 - P(di|s)))` = probability that subtopic s is covered by at least one selected document
- **xQuAD**: `Π (1 - P(di|s))` = probability that subtopic s is NOT covered by any selected document

**2. Mathematical Behavior**:
- **PM2**: Rewards adding documents that cover new subtopics OR reinforce already covered ones
- **xQuAD**: Rewards adding documents that cover previously uncovered subtopics MORE

**3. Greedy Selection Strategy**:
- **PM2**: More balanced between relevance and coverage reinforcement
- **xQuAD**: More aggressive about covering new subtopics

#### Practical Implications

**PM2 Characteristics**:
- Tends to select more relevant documents overall
- May select multiple documents for the same important subtopic
- Better when subtopic importance varies significantly
- More conservative diversity approach

**xQuAD Characteristics**:
- More aggressive diversification
- Strongly penalizes redundancy
- Better when all subtopics should be equally represented
- May sacrifice some relevance for broader coverage

#### When to Use Which

**Use PM2 when**:
- Subtopic importance varies significantly (some aspects much more important)
- Users might want multiple perspectives on the same subtopic
- Relevance is more critical than perfect coverage
- Query has a clear primary intent with secondary aspects

**Use xQuAD when**:
- All query subtopics are roughly equally important
- Perfect coverage is more important than redundancy
- Users want to see maximum variety
- Exploratory search scenarios

#### Example Comparison
For query "python" with subtopics: programming language (70%), snake species (20%), Monty Python (10%)

**PM2 might select**:
1. Python programming tutorial (high relevance)
2. Python data science guide (reinforces programming)
3. Python snake encyclopedia (covers snake subtopic)

**xQuAD might select**:
1. Python programming tutorial
2. Python snake encyclopedia  
3. Monty Python comedy sketches (ensures all subtopics covered)

The choice depends on whether you want depth in important topics (PM2) or breadth across all topics (xQuAD).

### Real-World Scenarios Where Each Excels

#### Scenario 1: PM2 is Better - Medical Search Query
**Query**: "heart disease treatment"
**Subtopics**: 
- Medications (60% importance)
- Surgery (25% importance) 
- Lifestyle changes (10% importance)
- Alternative medicine (5% importance)

**User Context**: Doctor looking for evidence-based treatment options for a patient

**Why PM2 Wins**:
- Medical professionals need authoritative, evidence-based information
- Multiple high-quality sources on medications are valuable (different drugs, studies, guidelines)
- Having 3 excellent medication papers is better than 1 medication + 1 alternative medicine
- Reinforcing critical subtopics with multiple perspectives is essential for life-critical decisions

**PM2 Results**:
1. "ACE Inhibitors for Heart Disease: Clinical Guidelines" (medications)
2. "Beta Blockers in Cardiovascular Treatment" (medications) 
3. "Cardiac Surgery Outcomes Study" (surgery)
4. "Exercise and Heart Health" (lifestyle)

**xQuAD Results**:
1. "ACE Inhibitors for Heart Disease: Clinical Guidelines" (medications)
2. "Cardiac Surgery Outcomes Study" (surgery)
3. "Exercise and Heart Health" (lifestyle)
4. "Herbal Remedies for Heart Conditions" (alternative medicine)

**Outcome**: PM2 provides better clinical utility by giving multiple medication options rather than forcing inclusion of less reliable alternative medicine content.

#### Scenario 2: xQuAD is Better - Travel Planning Query
**Query**: "things to do in Paris"
**Subtopics**:
- Museums/Culture (30%)
- Food/Restaurants (25%)
- Architecture/Landmarks (25%)
- Nightlife (10%)
- Shopping (10%)

**User Context**: Tourist planning a 5-day trip who wants to experience diverse aspects of Paris

**Why xQuAD Wins**:
- Tourist wants to discover all different aspects of the city
- One good museum recommendation is sufficient; seeing 3 museum articles is redundant
- Broader coverage helps plan a well-rounded itinerary
- Serendipitous discovery is more valuable than deep expertise in one area

**PM2 Results**:
1. "Ultimate Guide to the Louvre" (museums)
2. "Hidden Gems in Paris Museums" (museums)
3. "Best French Restaurants in Paris" (food)
4. "Notre Dame and Gothic Architecture" (architecture)

**xQuAD Results**:
1. "Ultimate Guide to the Louvre" (museums)
2. "Best French Restaurants in Paris" (food)
3. "Notre Dame and Gothic Architecture" (architecture)
4. "Paris Nightlife: Bars and Clubs Guide" (nightlife)

**Outcome**: xQuAD provides better trip planning utility by ensuring all aspects of the Paris experience are covered, helping the tourist create a more diverse and memorable itinerary.

#### Key Insight for Interviews
**PM2 excels when**:
- Domain expertise and authority matter more than variety
- Users need multiple high-quality sources on critical topics
- Subtopic importance is highly skewed
- Decisions have high stakes (medical, financial, legal)

**xQuAD excels when**:
- Discovery and exploration are the primary goals
- Users want to be surprised or learn about new areas
- All subtopics provide similar value
- Breadth of knowledge is more important than depth

#### MMR (Maximal Marginal Relevance)
**Formula**:
- **Simple**: 
```
MMR = arg max[λ × Sim1(Di,Q) - (1-λ) × max Sim2(Di,Dj)]
                Di∈R\S                    Dj∈S
```
- **LaTeX**: 
$\text{MMR} = \arg\max_{D_i \in R \setminus S} \left[\lambda \cdot \text{Sim}_1(D_i,Q) - (1-\lambda) \cdot \max_{D_j \in S} \text{Sim}_2(D_i,D_j)\right]$
Where:
- Sim1(Di,Q) = similarity between document Di and query Q
- Sim2(Di,Dj) = similarity between documents Di and Dj
- S = already selected documents
- λ balances relevance vs diversity

**Practice**: Implement MMR algorithm
```python
def mmr_diversify(query, docs, selected, lambda_param=0.5):
    max_mmr = -1
    best_doc = None
    
    for doc in docs:
        if doc in selected:
            continue
            
        relevance = similarity(doc, query)
        max_similarity = 0
        for sel_doc in selected:
            max_similarity = max(max_similarity, similarity(doc, sel_doc))
        
        mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
        
        if mmr_score > max_mmr:
            max_mmr = mmr_score
            best_doc = doc
    
    return best_doc
```

## Advanced Topics

### Neural Information Retrieval

#### DPR (Dense Passage Retrieval)
- **What it does**: Uses BERT-style encoders to create dense vector representations
- **Architecture**: Separate encoders for queries and passages
- **Training**: Contrastive learning with positive/negative passage pairs
- **Advantages**: Captures semantic similarity beyond exact keyword matching
- **Use case**: First-stage retrieval to replace traditional sparse methods like BM25

#### ColBERT (Contextualized Late Interaction over BERT)
- **What it does**: Efficient neural ranking with contextualized representations
- **Architecture**: Separate BERT encoders for queries and documents
- **Interaction**: Late interaction using MaxSim operation
- **Formula**: 
  - **Simple**: `Score(q,d) = Σ max(E_q[i] · E_d[j])` for each query token i
  - **LaTeX**: $\text{Score}(q,d) = \sum_{i \in q} \max_{j \in d} E_q[i] \cdot E_d[j]$
- **Advantages**: Better efficiency than cross-attention while maintaining quality
- **Use case**: Efficient re-ranking of retrieved candidates

### Modern Production Systems
- **Multi-Objective**: Balancing relevance, diversity, freshness, business metrics
- **Real-Time Learning**: Online learning from user interactions
- **Personalization**: User context, historical behavior, real-time signals
- **Scalability**: Distributed systems, approximate methods

### LLM Integration in Search/RecSys
- **Query Understanding**: Intent classification, query expansion
- **Content Enhancement**: Generate metadata, summaries, features
- **Result Generation**: Direct answer generation vs traditional ranking
- **Challenges**: Latency, hallucination, cost

### A/B Testing for Recommender Systems

![A/B Testing Framework](https://miro.medium.com/v2/resize:fit:1400/1*5btlaGf-ktVhV7mTQzd44A.png)

#### Network Effects Problem
**Challenge**: Users influence each other through shares, follows, viral content
- **Cross-contamination**: Treatment group content spreads to control group
- **Creator influence**: Algorithm changes affect creator behavior
- **Viral spillover**: Trending content crosses experimental boundaries

**Solutions**:
1. **Cluster Randomization**: Group by geography, social networks, or creator communities
2. **Temporal Separation**: Run treatment and control in different time periods
3. **Creator-level Randomization**: Keep creator-audience pairs in same group
4. **Content Isolation**: Separate trending algorithms for each group

#### Sample Size and Statistical Rigor
**Power Calculation Considerations**:
- **Minimum Detectable Effect (MDE)**: Smallest meaningful change to detect
- **Baseline variance**: Historical metric volatility
- **Multiple comparisons**: Bonferroni correction for multiple metrics
- **Practical significance**: Statistical vs business significance

**Formula**: `n ≈ 16σ²/δ²` where δ = MDE, σ = standard deviation

#### Metric Framework
**Primary Metrics**:
- Session length, completion rate, user engagement

**Secondary Metrics**: 
- Likes, shares, comments, follows, creator interactions

**Guardrail Metrics**:
- User retention and platform health
- Content diversity and creator fairness
- Time-spent upper bounds (healthy usage)
- Misinformation and harmful content rates

#### Harmful Behavior Detection
**Addiction Signals**:
- Session frequency × duration patterns
- Sleep pattern disruption indicators
- Social interaction displacement
- Compulsive usage patterns (rapid returns after closing app)

**Content Quality Monitoring**:
- Echo chamber formation (diversity scores)
- Misinformation propagation rates
- Harmful challenge spread
- Mental health impact indicators

**Implementation**:
```python
def detect_harmful_patterns(user_sessions):
    # Addiction pattern detection
    session_frequency = len([s for s in user_sessions if s.date == today])
    avg_session_length = mean([s.duration for s in user_sessions])
    rapid_returns = count_rapid_app_returns(user_sessions)
    
    # Content diversity scoring
    content_diversity = calculate_topic_entropy(user_sessions)
    
    # Threshold-based flagging
    addiction_score = weighted_score(session_frequency, avg_session_length, rapid_returns)
    health_score = content_diversity * engagement_quality
    
    return addiction_score, health_score
```