# ML System Design Interview Preparation Guide

## Table of Contents
1. [Feature Engineering](#feature-engineering)
2. [Training Pipeline](#training-pipeline)
3. [Problem Statement and Metrics](#problem-statement-and-metrics)
4. [System Architecture](#system-architecture)
5. [Scale and Performance](#scale-and-performance)
6. [Common ML System Design Patterns](#common-ml-system-design-patterns)
7. [Interview Framework](#interview-framework)
8. [Additional Topics](#additional-topics)

## Feature Engineering

![Feature Engineering Pipeline](https://developers.google.com/static/machine-learning/crash-course/images/categorical-vocabulary-concept.svg)
*Figure 3: Categorical feature encoding process from raw strings to feature vectors. Source: [Google Machine Learning Course](https://developers.google.com/machine-learning/crash-course/categorical-data/one-hot-encoding)*

### One Hot Encoding

**Problems:**
- Expansive computation and high memory consumption
- Creates high-dimensional feature vectors
- Not suitable for NLP tasks with large vocabularies

**Best Practices:**
- Group non-important classes into an "other" class
- Ensure pipeline can handle unseen data in test set
- Use `pandas.get_dummies()` or `sklearn.OneHotEncoder`
- **Tech Company Reality:** Not practical for large cardinality features (>1000 categories)

### Feature Hashing

Converts text data or high-cardinality categorical data into feature vectors of arbitrary dimensionality using hash functions.

**Benefits:**
- Handles high cardinality features (100-1000+ unique values)
- Hashing trick allows multiple values to be encoded as the same value
- Reduces dimensionality and memory usage
- **Trade-off:** Lower desired dimensionality = higher chance of collision

**Tech Company Usage:**
- **Popular at:** Booking.com, Meta, Yahoo, Yandex, Avazu, Criteo
- **Use cases:** Ad targeting, click prediction, large-scale recommendation systems

**Problems:**
- Hash collisions are common if hash size is too small
- High collision rate prevents model from differentiating between feature values
- Memory consumption if hash size is too large

### Crossed Features

**Definition:** Combination of two or more categorical variables (cardinality = c1 × c2)
- Usually paired with hashing trick to manage high dimensionality
- **Example:** Uber uses longitude × latitude crosses for location-based features

**Tech Company Usage:**
- **LinkedIn:** User location × job title for job recommendation models
- **Airbnb:** Cross features for search ranking models
- **Google:** Query × user location for search relevance

### Embeddings

**Purpose:** Capture semantic meaning of categorical features in dense vector representations

**Benefits:**
- **Example:** Word2Vec embeds words into dense multi-dimensional space
- Significantly improves prediction accuracy over sparse representations
- Captures relationships between categories

**Implementation:**
```python
tf.keras.layers.Embedding(vocab_size=1000, embedding_dim=5)
```

**Dimensionality Guidelines:**
- TensorFlow recommendation: `d = 4√D` where D = number of categories
- Usually determined experimentally or from domain experience
- Pre-compute and store in key-value storage to reduce inference latency

**Tech Company Usage:**
- **Twitter/X:** User embeddings for recommendations and nearest neighbor searches
- **DoorDash:** Store2Vec for personalized store feeds
  - Each store = one word, each user session = one sentence
  - Sum vectors for stores ordered in last 6 months/100 orders
  - Use cosine distance between store and consumer embeddings
- **Instagram:** Account embeddings for content recommendations

![Embedding vs One-Hot Encoding Comparison](https://www.researchgate.net/profile/Hongjie-Wang-31/publication/336910389/figure/fig4/AS:818842127994881@1572183297553/Training-comparison-of-one-hot-encoding-and-embedding-of-the-top-100-stations.png)
*Figure 4: Performance comparison between one-hot encoding and embedding approaches. Source: [ResearchGate](https://www.researchgate.net/figure/Training-comparison-of-one-hot-encoding-and-embedding-of-the-top-100-stations_fig4_336910389)*

### Numeric Features

**Normalization:** Scale data to [-1, 1] or [0, 1] range

**Standardization:**
- **Normal distribution:** `v = (v - v_mean) / v_std`
- **Power law distribution:** `v = log(1 + v) / (1 + v_median)`
- **Outlier handling:** Use clipping to prevent extreme values from skewing normalization

## Training Pipeline

### Data Partitioning

**File Formats:**
- **Parquet:** Optimized for write-once, read-many analytics
  - 30x faster than CSV
  - 99% cost reduction and data scan reduction
- **ORC (Optimized Row Columnar):** Better for read-heavy operations and Hive integration
- **Best Practice:** Partition by time for efficiency in time-series data

### Handling Imbalanced Class Distribution

#### 1. Class Weights in Loss Function
```python
# Example: 95% non-spam, 5% spam
loss = -w0 * y * log(p) - w1 * (1-y) * log(1-p)
```

#### 2. Resampling Techniques
- **Naive resampling:** Undersample majority class
- **SMOTE (Synthetic Minority Oversampling Technique):**
  - Synthesize minority class samples based on existing ones
  - Find k-nearest neighbors and interpolate
  - **Note:** Rarely used in production due to computational overhead

### Loss Function Selection

**Binary Classification:** Cross-entropy loss

**CTR Prediction:** Normalized cross-entropy loss (sensitive to background conversion rate)

**Forecasting:**
- **MAPE (Mean Absolute Percentage Error):**
  ```
  MAPE = (1/n) * Σ|At - Ft| / |At|
  ```
- **SMAPE (Symmetric Mean Absolute Percentage Error):**
  ```
  SMAPE = (100%/n) * Σ|Ft - At| / ((|At| + |Ft|)/2)
  ```

**Tech Company Examples:**
- **Uber:** RNN, Gradient Boosting Trees, SVR for demand forecasting
- **DoorDash:** Quantile loss for delivery demand forecasting
  ```
  L(y,ŷ) = max(τ(y-ŷ), (τ-1)(y-ŷ))
  ```

### Retraining Requirements

**Why Retrain:**
- Data distribution is non-stationary
- User behavior changes over time
- New trends and seasonal patterns emerge

**Frequency:**
- **AdTech/Recommendations:** Multiple times per day
- **General ML:** Weekly to monthly depending on domain

**Scheduling Tools:**
- **Apache Airflow:**
  - ✅ Good GUI, strong community support, independent scheduling
  - ❌ Less flexibility, difficult to manage massive pipelines
- **Luigi:**
  - ✅ Rich library ecosystem
  - ❌ Not very scalable, difficult to create/test tasks

## Problem Statement and Metrics

### Video Recommendation System (YouTube Example)

#### Problem Statement
Build a video recommendation system that maximizes user engagement while introducing content diversity.

#### Metrics Design

**Offline Metrics:**
- Precision@K, Recall@K
- Ranking loss (e.g., pairwise ranking loss)
- Log loss for probability calibration

**Online Metrics:**
- **Primary:** Click-through rate (CTR), watch time, session duration
- **Secondary:** Conversion rates, user retention, content diversity metrics
- **A/B Testing:** Statistical significance testing with proper power analysis

#### Requirements

**Training:**
- Handle viral content and temporal changes
- Retrain multiple times per day
- Balance model complexity with training time

**Inference:**
- Serve 100 video recommendations per homepage visit
- **Latency:** <200ms (ideally <100ms)
- **Exploration vs Exploitation:** Balance familiar content with discovery

## System Architecture

![Multi-Stage Recommendation Architecture](https://eugeneyan.com/assets/discovery-system-2x2.jpg)
*Figure 1: Two-stage architecture pattern for recommendation systems - splitting into offline/online environments and candidate retrieval/ranking steps. Source: [Eugene Yan](https://eugeneyan.com/writing/system-design-for-discovery/)*

### Multi-Stage Model Architecture

#### Stage 1: Candidate Generation

**Purpose:** Narrow down from millions of videos to hundreds of candidates

**Feature Engineering:**
- User watch history (video IDs, watch duration)
- User demographics and preferences
- Video metadata (category, upload time, popularity)

**Training Data:**
- User-video interaction matrix
- Balance between training time and model accuracy
- Typically use 1-6 months of recent data

**Model Choices:**
- **Traditional:** Matrix Factorization (collaborative filtering)
- **Modern Production:** 
  - Inverted index (Lucene, Elasticsearch)
  - Approximate nearest neighbor search (FAISS, Google ScaNN)
  - Two-tower neural networks

#### Stage 2: Ranking Model

**Purpose:** Rank and select final recommendations from candidates

**Feature Engineering:**
- User features (demographics, historical preferences)
- Video features (metadata, engagement metrics)
- Contextual features (time, device, location)
- Cross features (user-video interactions)

**Training Data:**
- Positive samples: videos user watched
- Negative samples: videos shown but not watched
- **Typical ratio:** 2% positive, 98% negative

**Model Architecture:**
- Start simple (logistic regression), increase complexity gradually
- **Deep Learning:** Fully connected neural networks (FCNN)
- **Activation:** ReLU for hidden layers, sigmoid for output
- **Loss:** Cross-entropy

### High-Level System Design

![ML System Architecture Overview](https://miro.medium.com/v2/resize:fit:1400/1*cgh8rlgMy3gGjdHbOvXfkg.png)
*Figure 2: High-level ML system architecture showing the flow from data collection to model serving. Source: [Louis Dorard](https://medium.com/louis-dorard/architecture-of-a-real-world-machine-learning-system-795254bec646)*

```
User Request → Load Balancer → Application Server
                                      ↓
              User Profile Service ← Feature Pipeline → Video Metadata Service
                                      ↓
              Candidate Generation Service → Ranking Service → Response
                                      ↓
                    Model Repository (S3/GCS)
```

**Key Components:**

1. **Databases:**
   - User watch history
   - Search query database
   - User/Video metadata
   - Historical recommendations

2. **Feature Pipeline:**
   - High throughput requirement (multiple daily retrains)
   - Use Spark, Elastic MapReduce, or Google Dataproc
   - Real-time and batch feature computation

3. **Model Repository:**
   - Store trained models (AWS S3, Google Cloud Storage)
   - Version control and model rollback capabilities

4. **Serving Infrastructure:**
   - Model-as-a-service in Docker containers
   - Kubernetes for auto-scaling
   - Load balancers for high availability

## Scale and Performance

### Capacity Estimation (YouTube Example)

**Assumptions:**
- Video views per month: 150 billion
- 10% from recommendations: 15 billion
- 100 recommendations per homepage visit
- 2% click-through rate
- 1.3 billion total users

**Data Size Calculation:**
- Positive labels: 15 billion/month
- Negative labels: 750 billion/month
- Features per sample: ~100 features × 500 bytes = 50KB per row
- **Total monthly data:** 500 bytes × 800 billion rows = 0.4 petabytes
- **Storage strategy:** Keep 6-12 months, archive older data to cold storage

**Bandwidth Requirements:**
- Peak: 10 million recommendation requests/second
- Each request ranks 1K-10K video candidates
- **QPS:** Handle millions of queries per second

### Scaling Strategies

**Horizontal Scaling:**
- Multiple application servers with load balancing
- Separate candidate generation and ranking services
- Database sharding by user ID or geographic region

**Performance Optimization:**
- Cache popular recommendations
- Pre-compute embeddings and store in key-value stores
- Use approximate algorithms for real-time serving
- Implement circuit breakers for fault tolerance

## Common ML System Design Patterns

### Real-Time vs Batch Processing

![ML Pipeline Architecture](https://neptune.ai/wp-content/uploads/2023/11/foreach-architecture.png)
*Figure 5: ML Pipeline leveraging DAG (Directed Acyclic Graph) and foreach pattern for efficient workflow management. Source: [Neptune.ai](https://neptune.ai/blog/ml-pipeline-architecture-design-patterns)*

**Real-Time (Stream Processing):**
- **Use cases:** Fraud detection, real-time recommendations
- **Technologies:** Apache Kafka, Apache Storm, Google Dataflow
- **Latency:** <100ms
- **Trade-off:** Lower accuracy for speed

**Batch Processing:**
- **Use cases:** Training data preparation, feature engineering
- **Technologies:** Apache Spark, Hadoop MapReduce
- **Latency:** Hours to days
- **Trade-off:** Higher accuracy, more computational resources

### Model Serving Patterns

#### 1. Model-as-a-Service
- Models deployed in containers (Docker/Kubernetes)
- REST API endpoints for predictions
- Easy to scale and version

#### 2. Embedded Models
- Models compiled into application code
- Ultra-low latency
- Harder to update and version

#### 3. Edge Deployment
- Models deployed on mobile devices or edge servers
- Offline capability
- Privacy benefits

### A/B Testing Framework

![A/B Testing Architecture](https://www.alibabacloud.com/blog/basic-concepts-and-architecture-of-a-recommender-system_596642)
*Figure 6: Enterprise recommendation system architecture showing matching and ranking modules. Source: [Alibaba Cloud](https://www.alibabacloud.com/blog/basic-concepts-and-architecture-of-a-recommender-system_596642)*

**Statistical Considerations:**
- Power analysis for sample size determination
- Multiple testing correction (Bonferroni, FDR)
- Minimum detectable effect size

**Metrics:**
- **Primary:** Business metrics (revenue, engagement)
- **Secondary:** Technical metrics (latency, error rates)
- **Guardrail:** Safety metrics (user satisfaction, fairness)

## Interview Framework

![ML System Design Interview Flow](https://www.researchgate.net/profile/M-Abdel-Latif/publication/348613622/figure/fig1/AS:981827983134720@1611123893020/The-overall-recommendation-system-architecture.png)
*Figure 8: Overall recommendation system architecture showing the complete flow from user input to recommendations. Source: [ResearchGate](https://www.researchgate.net/figure/The-overall-recommendation-system-architecture_fig1_348613622)*

### 1. Problem Clarification (5-10 minutes)
- **Scope:** What exactly are we trying to predict/recommend?
- **Scale:** How many users, items, requests per second?
- **Constraints:** Latency requirements, budget, existing infrastructure
- **Success metrics:** How do we measure success?

### 2. High-Level Design (10-15 minutes)
- **Data flow:** User request → candidate generation → ranking → response
- **Major components:** Databases, services, ML pipeline
- **API design:** Request/response format
- **Technology choices:** Justify major architectural decisions

### 3. Deep Dive (15-20 minutes)
Choose 2-3 areas to dive deep:
- **Feature engineering:** What features to use and why
- **Model architecture:** Algorithm choice and justification
- **Training pipeline:** Data processing, model training, evaluation
- **Serving infrastructure:** How to serve predictions at scale

### 4. Scale and Advanced Topics (10-15 minutes)
- **Bottlenecks:** Identify and address system bottlenecks
- **Monitoring:** What metrics to track and alert on
- **Failure handling:** What happens when components fail
- **Advanced features:** Personalization, cold start, fairness

## Additional Topics

### Cold Start Problem

**User Cold Start:**
- New users with no historical data
- **Solutions:** Demographics-based recommendations, popular content, onboarding surveys

**Item Cold Start:**
- New videos/products with no engagement data
- **Solutions:** Content-based features, creator popularity, similar item recommendations

**System Cold Start:**
- Entirely new recommendation system
- **Solutions:** Transfer learning, domain adaptation, hybrid approaches

### Monitoring and Observability

**Model Performance:**
- **Drift detection:** Statistical tests for feature and prediction drift
- **Model decay:** Track online vs offline metric correlation
- **Shadow testing:** Run new models alongside production without serving results

**System Health:**
- **Latency:** P50, P95, P99 response times
- **Throughput:** Requests per second, successful predictions
- **Error rates:** 4xx/5xx errors, model prediction failures
- **Resource utilization:** CPU, memory, GPU usage

### Fairness and Bias

**Types of Bias:**
- **Historical bias:** Training data reflects past inequities
- **Representation bias:** Underrepresented groups in training data
- **Measurement bias:** Different quality of data across groups

**Mitigation Strategies:**
- **Pre-processing:** Data augmentation, synthetic data generation
- **In-processing:** Fairness constraints in loss function
- **Post-processing:** Adjust predictions to ensure fairness metrics

**Metrics:**
- **Individual fairness:** Similar individuals receive similar treatment
- **Group fairness:** Equal outcomes across demographic groups
- **Equalized odds:** Equal true positive rates across groups

### Privacy and Security

**Privacy Techniques:**
- **Differential privacy:** Add noise to preserve individual privacy
- **Federated learning:** Train models without centralizing data
- **Data anonymization:** Remove or hash personally identifiable information

**Security Considerations:**
- **Model extraction attacks:** Prevent unauthorized model copying
- **Adversarial examples:** Robust training against malicious inputs
- **Data poisoning:** Validate training data integrity

### Model Versioning and Deployment

**Deployment Strategies:**
- **Blue-green deployment:** Maintain two identical production environments
- **Canary deployment:** Gradually roll out to small percentage of traffic
- **Shadow deployment:** Run new model alongside current without serving

**Version Control:**
- **Model artifacts:** Store models with metadata (training data, hyperparameters)
- **Feature pipelines:** Version feature transformation code
- **Rollback strategy:** Quick revert to previous model version

### Advanced Architecture Patterns

#### Lambda Architecture
- **Batch layer:** Historical data processing for accuracy
- **Speed layer:** Real-time data processing for low latency
- **Serving layer:** Merge batch and real-time views

![Lambda Architecture Diagram](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/images/ml-lifecycle-arch.png)
*Figure 7: ML lifecycle architecture showing different phases and components. Source: [AWS Well-Architected ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/ml-lifecycle-architecture-diagram.html)*

#### Kappa Architecture
- **Stream-only:** Process all data as streams
- **Simpler:** Single pipeline for batch and real-time
- **Trade-off:** May sacrifice some accuracy for simplicity

### Cost Optimization

**Compute Costs:**
- **Training:** Use spot instances, schedule during off-peak hours
- **Inference:** Auto-scaling based on traffic patterns
- **Storage:** Tiered storage (hot/warm/cold) based on access patterns

**Model Efficiency:**
- **Model compression:** Pruning, quantization, distillation
- **Approximate algorithms:** Trade accuracy for speed/cost
- **Caching:** Store frequent predictions to reduce compute

### Follow-Up Questions & Advanced Topics

#### Handling Concept Drift
- **Detection:** Monitor feature distributions and model performance
- **Adaptation:** Incremental learning, ensemble methods
- **Solution:** Bayesian logistic regression for online updates

#### Exploration vs Exploitation
- **Multi-armed bandits:** Balance trying new content vs serving known good content
- **Implementation:** 
  - 98% exploitation (ranked recommendations)
  - 2% exploration (random or uncertainty-based sampling)
- **Advanced:** Contextual bandits, Thompson sampling

#### Real-Time Feature Engineering
- **Stream processing:** Apache Kafka + Apache Flink/Storm
- **Feature stores:** Centralized feature serving (Feast, Tecton)
- **Consistency:** Ensure training/serving feature parity

#### Model Interpretability
- **SHAP values:** Understand feature importance for individual predictions
- **LIME:** Local explanations for complex models
- **Business requirement:** Regulatory compliance, user trust

### Common Interview Questions

1. **"How would you design a recommendation system for [company]?"**
   - Follow the framework: clarify → design → deep dive → scale

2. **"How do you handle training data that's biased?"**
   - Discuss bias types, detection methods, mitigation strategies

3. **"Your model performance is degrading in production. How do you debug?"**
   - Feature drift, data quality, model staleness, infrastructure issues

4. **"How do you ensure your ML system is reliable?"**
   - Monitoring, alerting, fallback mechanisms, gradual deployments

5. **"How would you A/B test a new recommendation algorithm?"**
   - Experimental design, metrics selection, statistical significance

### Preparation Tips

1. **Practice drawing systems:** Get comfortable with architectural diagrams
2. **Know the numbers:** Memorize common capacity estimation figures
3. **Understand trade-offs:** Every design decision has pros and cons
4. **Think end-to-end:** From data collection to model serving to business impact
5. **Ask clarifying questions:** Scope the problem before jumping into solutions
6. **Consider failure modes:** What happens when things go wrong?

### Key Metrics to Remember

- **Latency targets:** <100ms for real-time, <200ms acceptable
- **Availability:** 99.9% uptime (8.77 hours downtime/year)
- **Throughput:** Think in terms of QPS (queries per second)
- **Storage:** Understand data size calculations (features × samples × retention)
- **Cost:** Training costs vs serving costs vs storage costs

---

**Remember:** ML system design interviews test your ability to think systematically about large-scale machine learning systems. Focus on demonstrating your understanding of trade-offs, scalability considerations, and real-world constraints rather than just algorithmic knowledge.