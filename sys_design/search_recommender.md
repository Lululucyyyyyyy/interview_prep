
## Common Algorithm Questions & Solutions

### 1. "Implement a basic search ranking algorithm"
```python
def search_rank(query, documents, algorithm='bm25'):
    if algorithm == 'bm25':
        scores = [bm25_score(query.split(), doc, documents) for doc in documents]
    elif algorithm == 'tfidf':
        scores = [tfidf_score(query.split(), doc, documents) for doc in documents]
    
    # Sort by score descending
    ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked_docs]
```

### 2. "How would you detect and handle duplicate documents?"
```python
def detect_duplicates(documents, threshold=0.8):
    # Use MinHash for efficient duplicate detection
    duplicates = []
    for i, doc1 in enumerate(documents):
        for j, doc2 in enumerate(documents[i+1:], i+1):
            similarity = jaccard_similarity(doc1, doc2)
            if similarity > threshold:
                duplicates.append((i, j, similarity))
    return duplicates

def jaccard_similarity(doc1, doc2):
    set1, set2 = set(doc1.split()), set(doc2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0
```

### 3. "Implement query expansion using word embeddings"
```python
def expand_query(query, word_vectors, top_k=3):
    expanded_terms = []
    for term in query.split():
        if term in word_vectors:
            similar_words = word_vectors.most_similar(term, topn=top_k)
            expanded_terms.extend([word for word, score in similar_words if score > 0.7])
    return query + " " + " ".join(expanded_terms)
```

### 4. "Design A/B test for evaluating search ranking changes"
```python
def ab_test_design(test_name, traffic_split=0.5, metrics=['ctr', 'session_length']):
    return {
        'test_name': test_name,
        'control_group': {'traffic': 1 - traffic_split, 'algorithm': 'current'},
        'treatment_group': {'traffic': traffic_split, 'algorithm': 'new'},
        'metrics': metrics,
        'minimum_sample_size': calculate_sample_size(),
        'duration_days': 14,
        'significance_level': 0.05
    }

def calculate_sample_size(baseline_rate=0.1, min_detectable_effect=0.02, power=0.8):
    # Statistical calculation for sample size
    # Simplified version - use proper statistical libraries in practice
    return int(16 * (baseline_rate * (1 - baseline_rate)) / (min_detectable_effect ** 2))
```

### 5. "Implement real-time personalization for recommendations"
```python
class RealTimePersonalizer:
    def __init__(self):
        self.user_profiles = {}
        self.decay_factor = 0.9
    
    def update_profile(self, user_id, item_id, interaction_type, timestamp):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        
        # Apply time decay to existing preferences
        current_time = time.time()
        for item, (score, last_update) in self.user_profiles[user_id].items():
            time_diff = current_time - last_update
            decay = self.decay_factor ** (time_diff / 3600)  # Hourly decay
            self.user_profiles[user_id][item] = (score * decay, last_update)
        
        # Update with new interaction
        weight = self.get_interaction_weight(interaction_type)
        current_score = self.user_profiles[user_id].get(item_id, (0, timestamp))[0]
        new_score = current_score + weight
        self.user_profiles[user_id][item_id] = (new_score, timestamp)
    
    def get_interaction_weight(self, interaction_type):
        weights = {'view': 1, 'like': 3, 'share': 5, 'purchase': 10}
        return weights.get(interaction_type, 1)
```

## Technical Deep Dives

### 1. "Explain the mathematical foundation of PageRank"
PageRank models the probability that a random surfer will arrive at a page. The algorithm:

**Formula**: `PR(A) = (1-d)/N + d × Σ(PR(Ti)/C(Ti))`

Where:
- d = damping factor (typically 0.85)
- N = total number of pages
- Ti = pages that link to page A
- C(Ti) = number of outbound clicks from page Ti

**Matrix Form**: `PR = (1-d)/N × e + d × M × PR`

**Power Iteration Method**:
```python
def pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    N = len(graph)
    pr = np.ones(N) / N
    
    for _ in range(max_iter):
        new_pr = np.zeros(N)
        for i in range(N):
            for j in graph[i]:  # j links to i
                new_pr[i] += pr[j] / len(graph[j])
        
        new_pr = (1 - damping) / N + damping * new_pr
        
        if np.linalg.norm(new_pr - pr) < tol:
            break
        pr = new_pr
    
    return pr
```

**Key Insights**:
- Handles link spam better than simple link counting
- Dampening factor prevents rank sinks
- Convergence guaranteed for strongly connected graphs
- Can be personalized for user-specific rankings

### 2. "How do you handle the vocabulary mismatch problem in IR?"
The vocabulary mismatch occurs when users and documents use different terms for the same concept.

**Solutions**:

1. **Query Expansion**:
   - Synonym dictionaries (WordNet)
   - Pseudo-relevance feedback
   - Word embeddings (Word2Vec, GloVe)
   
2. **Document Expansion**:
   - Add related terms to documents
   - Use external knowledge bases
   
3. **Semantic Matching**:
   - Dense retrieval (DPR, ColBERT)
   - Cross-encoder models
   - Latent semantic analysis

**Implementation Example**:
```python
def handle_vocabulary_mismatch(query, documents):
    # Method 1: Query expansion
    expanded_query = expand_with_synonyms(query)
    
    # Method 2: Semantic embedding
    query_embedding = encode_text(query)
    doc_embeddings = [encode_text(doc) for doc in documents]
    semantic_scores = [cosine_similarity(query_embedding, doc_emb) 
                      for doc_emb in doc_embeddings]
    
    # Method 3: Hybrid approach
    lexical_scores = [bm25_score(expanded_query, doc) for doc in documents]
    final_scores = [0.7 * lex + 0.3 * sem 
                   for lex, sem in zip(lexical_scores, semantic_scores)]
    
    return final_scores
```

### 3. "Describe the training process for a learning-to-rank model"
Learning-to-rank training involves several key steps:

**Data Preparation**:
```python
def prepare_ltr_data(queries, documents, relevance_labels):
    training_data = []
    for query in queries:
        # Get candidate documents for this query
        candidates = retrieve_candidates(query, documents)
        
        for doc in candidates:
            # Extract features
            features = extract_features(query, doc)
            # Get relevance label (0-4 scale typically)
            label = relevance_labels.get((query.id, doc.id), 0)
            
            training_data.append({
                'qid': query.id,
                'features': features,
                'label': label,
                'doc_id': doc.id
            })
    
    return training_data
```

**Training Process**:
1. **Feature Engineering**: Extract query-document features
2. **Loss Function**: Choose appropriate loss (pointwise, pairwise, listwise)
3. **Optimization**: Train model to minimize ranking loss
4. **Validation**: Use ranking metrics (NDCG, MAP) for evaluation

**Pairwise Training Example**:
```python
def train_pairwise_ranker(training_data):
    model = RankingModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for query_group in group_by_query(training_data):
            # Create all possible pairs
            pairs = create_preference_pairs(query_group)
            
            for (doc1, doc2, preference) in pairs:
                score1 = model(doc1.features)
                score2 = model(doc2.features)
                
                # Pairwise hinge loss
                if preference == 1:  # doc1 > doc2
                    loss = max(0, 1 - (score1 - score2))
                else:  # doc2 > doc1
                    loss = max(0, 1 - (score2 - score1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

### 4. "Explain cold start problem solutions in detail"
Cold start is a fundamental challenge in recommendation systems:

**Types of Cold Start**:

1. **New User Cold Start**:
```python
def handle_new_user_cold_start(user_demographics, popular_items):
    # Demographic-based recommendations
    age_group = categorize_age(user_demographics['age'])
    location = user_demographics['location']
    
    # Get popular items for similar demographic
    demographic_popular = get_popular_by_demographic(age_group, location)
    
    # Quick onboarding - ask for preferences on popular items
    onboarding_items = select_diverse_items(popular_items, k=20)
    
    return {
        'initial_recs': demographic_popular[:10],
        'onboarding_items': onboarding_items
    }
```

2. **New Item Cold Start**:
```python
def handle_new_item_cold_start(item, similar_items_index):
    # Content-based similarity
    content_features = extract_content_features(item)
    similar_items = find_similar_by_content(content_features, similar_items_index)
    
    # Transfer learning from similar items
    initial_users = []
    for similar_item in similar_items:
        users_who_liked = get_users_by_item(similar_item.id)
        initial_users.extend(users_who_liked)
    
    # Exploration strategy - show to diverse user segments
    exploration_users = select_diverse_users(initial_users)
    
    return exploration_users
```

3. **System Cold Start**:
```python
def handle_system_cold_start():
    # Use external data sources
    strategies = [
        'import_popular_items_from_similar_domains',
        'use_demographic_stereotypes',
        'implement_content_based_fallback',
        'create_editorial_recommendations'
    ]
    
    # Multi-armed bandit for strategy selection
    strategy_performance = {}
    for strategy in strategies:
        # Track performance and allocate traffic accordingly
        epsilon_greedy_selection(strategy, strategy_performance)
```

### 5. "How do you optimize for multiple objectives in ranking?"
Multi-objective optimization is crucial in production systems:

**Approaches**:

1. **Weighted Linear Combination**:
```python
def multi_objective_score(relevance, diversity, freshness, business_value, weights):
    return (weights['relevance'] * relevance + 
            weights['diversity'] * diversity + 
            weights['freshness'] * freshness + 
            weights['business'] * business_value)
```

2. **Pareto Optimization**:
```python
def pareto_ranking(candidates, objectives):
    pareto_front = []
    for candidate in candidates:
        is_dominated = False
        for other in candidates:
            if dominates(other, candidate, objectives):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(candidate)
    return pareto_front

def dominates(a, b, objectives):
    # a dominates b if a is better or
```
## System Design Examples

### Facebook News Feed
**Requirements:**
- Serve personalized content to 3B+ users
- Real-time updates and interactions
- Multiple content types (posts, ads, videos)
- High engagement and retention

**Architecture:**
1. **Content Ingestion**: User posts, page updates, ads
2. **Candidate Generation**: 
   - Friend activity (posts, likes, comments)
   - Page subscriptions and interests
   - Trending content and viral posts
3. **Ranking**: 
   - Engagement prediction (likes, comments, shares, time spent)
   - Content quality scoring
   - Recency and relevance
   - Personalization based on user history
4. **Diversity**: Ensure mix of content types and sources
5. **Serving**: Real-time API with caching layers

**Key Considerations:**
- Cold start for new users (demographic-based, popular content)
- Real-time signals (recent interactions, online friends)
- Anti-patterns (clickbait detection, misinformation filtering)
- Business objectives (ad revenue, user engagement)

### YouTube (Long-form Video)
**Requirements:**
- Recommend 30+ minute videos
- Optimize for watch time and session length
- Handle diverse content catalog
- Support creator discovery

**Architecture:**
1. **Candidate Generation**:
   - Collaborative filtering (similar users/channels)
   - Content-based (video metadata, categories)
   - Trending and viral content
2. **Ranking**:
   - Watch time prediction models
   - User engagement signals (likes, comments, subscriptions)
   - Video quality indicators (retention curves, completion rates)
   - Freshness vs evergreen content balance
3. **Multi-objective Optimization**:
   - Watch time (primary)
   - User satisfaction surveys
   - Creator ecosystem health
   - Content diversity

**Unique Challenges:**
- Long content consumption patterns
- Creator monetization considerations
- Content moderation at scale
- Handling seasonal/trending content

### TikTok (Short-form Video)
**Requirements:**
- Endless scroll of 15-60 second videos
- Maximize user session time
- Rapid content discovery
- Real-time trend detection

**Architecture:**
1. **Candidate Generation**:
   - For You Page algorithm
   - Audio/music-based recommendations
   - Hashtag and challenge-based discovery
   - Creator following
2. **Ranking**:
   - Completion rate prediction (critical for short videos)
   - Interaction prediction (likes, shares, comments)
   - Re-watch probability
   - Video quality and production value
3. **Real-time Features**:
   - Immediate feedback incorporation
   - Trending detection and amplification
   - Geographic and cultural relevance

**Unique Challenges:**
- Extremely short feedback loops
- Content velocity and virality
- Cultural and regional differences
- Creator economy and monetization

### Google Search Engine
**Requirements:**
- Handle billions of queries daily
- Sub-second response times
- Comprehensive web coverage
- High relevance and quality

**Architecture:**
1. **Crawling & Indexing**:
   - Web crawling at massive scale
   - Content processing and feature extraction
   - Inverted index construction
   - Real-time index updates
2. **Query Processing**:
   - Query understanding and intent detection
   - Spell correction and suggestion
   - Query expansion and rewriting
3. **Ranking**:
   - Multiple ranking signals (200+ factors)
   - PageRank and link analysis
   - Content quality and E-A-T signals
   - User behavior signals
   - Freshness and recency
4. **Serving**:
   - Distributed serving infrastructure
   - Result snippets and rich features
   - Personalization layer

**Key Challenges:**
- Spam and low-quality content detection
- Handling ambiguous queries
- Balancing algorithmic and business interests
- International and multilingual support