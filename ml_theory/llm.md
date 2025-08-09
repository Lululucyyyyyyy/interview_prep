### Long-Context Solutions

| Approach | Method | Memory Complexity | Compute Complexity | Context Length | Trade-offs |
|----------|--------|-------------------|-------------------|----------------|------------|
| **Efficient Attention** | Flash Attention | O(s) | O(s²) | Moderate extension | Implementation complexity |
| **Sparse Attention** | Sliding window, sparse patterns | O(s) | O(s×window) | Good extension | May miss dependencies |
| **Linear Attention** | Approximation methods | O(s) | O(s) | Excellent | Approximation quality |
| **State Space Models** | Mamba, S4 | O(s) | O(s) | Unlimited | Different training paradigm |
| **Retrieval-based** | RAG, memory systems | O(s) | O(s×retrieved) | Unlimited | Retrieval quality dependency |

### Inference Scaling Strategies

| Strategy | Type | How It Works | Advantages | Disadvantages | Best For |
|----------|------|--------------|------------|---------------|----------|
| **Best-of-N** | Parallel | Generate N, select best | Simple, effective | Expensive, over-optimization risk | High-stakes tasks |
| **Voting** | Parallel | Majority vote on answers | Robust to errors | Only works for discrete answers | Multiple-choice, math |
| **Tree Search** | Sequential | Explore reasoning paths | Systematic exploration | Exponential complexity | Complex reasoning |
| **Long CoT** | Sequential | Extended reasoning | Rich intermediate steps | Length constraints | Open-ended problems |
| **MBR** | Parallel | Consensus-based selection | Flexible utility functions | Requires good utility function | Quality-focused tasks |

### Multimodal Architecture Comparison

| Architecture | Vision Component | Integration Method | Strengths | Weaknesses | Examples |
|--------------|------------------|-------------------|-----------|------------|----------|
| **CLIP-style** | ViT encoder | Contrastive pre-training | Strong zero-shot | Limited fine-grained reasoning | CLIP, ALIGN |
| **VLM** | Frozen vision encoder | Linear projection to LLM | Leverages LLM capabilities | Vision encoder limitations | LLaVA, InstructBLIP |
| **End-to-end** | Trainable vision | Joint training | Optimal integration | Expensive training | Flamingo, DALL-E |
| **Multimodal Transformer** | Patch embeddings | Unified transformer | Native multimodal | Complex training | ViLT, METER |# LLM Theory Interview Preparation Guide

*Based on CS11-711 Advanced NLP Course Notes*

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Neural Text Representation](#neural-text-representation)
3. [Language Modeling](#language-modeling)
4. [Attention and Transformers](#attention-and-transformers)
5. [Pretraining](#pretraining)
6. [Inference and Generation](#inference-and-generation)
7. [Fine-tuning and Adaptation](#fine-tuning-and-adaptation)
8. [Reinforcement Learning](#reinforcement-learning)
9. [Evaluation](#evaluation)
10. [Advanced Topics](#advanced-topics)

---

## Fundamentals

### What is NLP and why do we need it?

**Definition**: NLP is technology that allows computers to process, generate, and interact with language such as text.

**Key Aspects**:
- **Learn useful representations**: Capture meaning in structured way (e.g., embeddings for classification)
- **Generate language**: Create text/code for dialogue, translation, QA
- **Bridge language and action**: Use language to perform tasks and interact with environments

**Common NLP System Building Methods**:
- **Rules**: Manual creation of rules
- **Prompting**: Using language model without training
- **Fine-tuning**: Machine learning from paired data `<X, Y>`

### Rule-based vs ML-based Systems

**Rule-based System Process**:
1. **Feature extraction**: Extract salient features from text (`h = f(x)`)
2. **Score calculation**: Calculate scores (`s = w · h`)
3. **Decision function**: Choose output (`ŷ = g(s)`)

| Aspect | Rule-based Systems | ML-based Systems |
|--------|-------------------|------------------|
| **Development** | Manual rule creation | Learning algorithm from data |
| **Word Variations** | Struggles with conjugations | Handles variations through embeddings |
| **Similarity** | Cannot share strength between similar words | Learns word similarities |
| **Negation** | Difficult to handle complex negation | Can learn negation patterns |
| **Metaphor/Analogy** | Very challenging | Better at non-literal language |
| **Language Support** | Requires separate rules per language | Can transfer across languages |
| **Data Requirements** | No training data needed | Requires training data |
| **Interpretability** | Highly interpretable | Less interpretable |
| **Performance Guarantees** | No guarantees without extensive testing | Improves with more data |

---

## Neural Text Representation

### Tokenization and Subword Models

**Core Problem**: Mapping text into sequence of discrete tokens from vocabulary `𝒱`

**Desirable Vocabulary Properties**:
- **Expressive**: Can represent any text (English, Japanese, code, etc.)
- **Efficient**: Not too large (more parameters) or too small (longer inputs)

### Tokenization Methods

| Method | Description | Pros | Cons | Use Case |
|--------|-------------|------|------|----------|
| **UTF-8** | Tokenize as UTF-8 bytes | Universal for any Unicode | Very long sequences, inefficient | Limited practical use |
| **Word-level** | Split on whitespace/punctuation | Simple, interpretable | Large vocabulary, OOV issues | Small-scale applications |
| **BPE** | Merge most common token pairs | Balanced vocab size, handles OOV | Training data dependent, multiple segmentations | Most modern LLMs |
| **SentencePiece** | Unicode-based with byte fallback | Language agnostic, robust | More complex setup | Multilingual models |

**BPE Considerations**:

| Issue | Problem | Solution |
|-------|---------|----------|
| **Training Data Dependency** | Under-represented languages get longer sequences | Upsample under-represented languages |
| **Multiple Segmentations** | Ambiguity in word splitting | Subword regularization during training |
| **Vocabulary Coverage** | Different domains need different tokens | Domain-specific vocabulary mixing |

### Word Embeddings

**Evolution**: One-hot vectors (sparse) → Continuous embeddings (dense vectors in `ℝᵈᵉᵐᵇ`)

**Embedding Layer**: Matrix where each row/column corresponds to vocabulary token

**Continuous Bag of Words (CBoW)**:
- Words that are similar are close in vector space
- Each vector element represents a feature
- Enables sharing strength among similar words

### Neural Network Features

**Deep CBoW**:
- Introduces hidden layers and nonlinearities (e.g., `tanh`)
- Learns complex patterns and feature combinations
- Without activation functions, stacking matrices collapses to single linear transformation

**Combination Features**: Handles patterns like "I don't love this movie" that simple BoW misses

---

## Language Modeling

### What is a Language Model?

**Definition**: A **probability distribution over all sequences** `P(X)`

**Applications**:
- **Score sequences**: Higher probability for natural sentences
- **Generate sequences**: Sample `x̂ ~ P(X)`
- **Conditional generation**: Generate continuation given context
- **Answer questions**: Score multiple-choice or generate answers
- **Classify text**: Score text conditioned on labels
- **Grammar correction**: Score/replace words

### Auto-regressive Language Models

**Key Idea**: Decompose sequence modeling into **next-token modeling**
```
P(X) = ∏(t=1 to T) P(xₜ | x₁, ..., xₜ₋₁)
```

### Model Types

### Language Model Evolution

| Model Type | Context | Strengths | Limitations | Training Method |
|------------|---------|-----------|-------------|-----------------|
| **Bigram** | Previous token only | Simple, fast | Very limited context | Count-based MLE |
| **N-gram** | N-1 previous tokens | Longer context than bigram | No parameter sharing, fixed context | Count-based with smoothing |
| **Feedforward Neural** | Fixed window | Learned embeddings, parameter sharing | Fixed context window | Gradient descent |
| **RNN** | Unlimited (in theory) | Variable length sequences | Vanishing gradients, sequential training | Backprop through time |
| **Transformer** | Full sequence | Parallel training, long dependencies | Quadratic memory/compute | Self-attention |

### N-gram vs Neural Language Model Comparison

| Aspect | N-gram Models | Neural Language Models |
|--------|---------------|------------------------|
| **Parameter Sharing** | No sharing between similar words | Shared embeddings for similar words |
| **Context Handling** | Fixed window, no intervening words | Flexible context, handles gaps |
| **Long Dependencies** | Cannot model beyond N-1 | Better long-range modeling |
| **Training Speed** | Extremely fast (counting) | Slower (gradient descent) |
| **Memory Usage** | Large for high N | Compact representations |
| **Generalization** | Poor to unseen n-grams | Better generalization |
| **When to Use** | Fast inference, perfect memorization needed | Most modern applications |

### Training Neural Networks

**Loss Functions**:
- **Binary Cross Entropy**: `L_BCE = −y log(p) − (1 − y)log(1 − p)`
- **Multi-class Cross Entropy**: `L_CE = −∑yᵢ log(pᵢ)`

**Optimization**:
- **Standard SGD**: `θₜ = θₜ₋₁ - ηgₜ`
- **Adam**: Most standard in NLP, considers rolling averages and momentum

**Evaluation Metrics**:
- **Log-likelihood**: Sum of log probabilities
- **Perplexity**: `PPL = 2^H = e^(-WLL)` (lower is better)

---

## Attention and Transformers

### RNNs and Their Limitations

### RNN Architectures Comparison

| Architecture | Gates | Memory | Strengths | Weaknesses | Best For |
|--------------|-------|--------|-----------|------------|----------|
| **Vanilla RNN** | None | Hidden state | Simple, interpretable | Vanishing gradients | Short sequences |
| **GRU** | Update, Reset | Hidden state | Fewer parameters than LSTM | Less flexible than LSTM | Medium sequences |
| **LSTM** | Forget, Input, Output | Cell state + Hidden state | Strong long-term memory | More parameters, slower | Long sequences |

### Attention Score Functions

| Function | Formula | Pros | Cons | Common Use |
|----------|---------|------|------|------------|
| **Nonlinear** | `w₂ᵀ tanh(W₁[q;k])` | Most expressive | Slower, more parameters | Early attention models |
| **Bilinear** | `qᵀWk` | Good balance | Moderate parameters | General purpose |
| **Dot Product** | `qᵀk` | Very fast, no parameters | Can be unstable with large dims | When dimensions match |
| **Scaled Dot Product** | `qᵀk / √dₖ` | Fast, stable | Simple | Transformer standard |

### Transformer Architecture

**Core Formula**: `Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V`

### Positional Encoding Methods

| Method | Type | Advantages | Disadvantages | Best For |
|--------|------|------------|---------------|----------|
| **Sinusoidal** | Fixed | No training needed, deterministic | Fixed patterns, limited flexibility | Original Transformer |
| **Learned Absolute** | Trainable | Flexible, task-adaptive | Cannot extrapolate to longer sequences | Fixed-length tasks |
| **Relative** | Computed | Models relative distances | More complex computation | Variable-length tasks |
| **RoPE** | Rotary | Extrapolates well, relative positioning | Complex implementation | Modern LLMs |

### Transformer Improvements

| Improvement | Original | Improved | Benefit | Used In |
|-------------|----------|----------|---------|---------|
| **Layer Norm Position** | Post-layer (after attention) | Pre-layer (before attention) | Better gradient flow | Most modern models |
| **Normalization** | LayerNorm | RMSNorm | Simpler, faster | LLaMA, others |
| **Attention Pattern** | Full multi-head | Grouped-query attention | Memory efficient, faster inference | LLaMA 2, others |
| **Activation** | ReLU | SwiGLU, GeGLU | Better performance | Recent models |

**Transformer vs RNN Comparison**:

| Feature | RNN | Transformer |
|---------|-----|-------------|
| Training Parallelization | Difficult | Easy |
| Computation Complexity | O(Td²) | O(T²d) |
| Memory (Inference) | O(1) | O(T²) |
| Long-range Dependencies | Struggles | Handled by attention |

---

## Pretraining

### Why Pretrain?

**Benefits**:
- **Transfer learning**: Apply knowledge from one task to another
- **Less task data**: Need less data for given performance
- **Better performance**: Higher than training from scratch
- **One model, multiple tasks**: Convenient and cost-effective

### Pretraining Objectives Comparison

| Objective | Model Examples | Training Style | Strengths | Weaknesses | Best For |
|-----------|----------------|----------------|-----------|------------|----------|
| **MLM (Masked LM)** | BERT, RoBERTa | Bidirectional | Better understanding tasks | Cannot generate naturally | Fine-tuning tasks |
| **ALM (Auto-regressive)** | GPT series, LLaMA | Left-to-right | Natural generation | Only left context | Generation, prompting |
| **Prefix LM** | T5, PaLM | Bidirectional prefix + autoregressive | Best of both worlds | More complex training | Versatile applications |
| **GLM** | ChatGLM | Blanks filling | Unified framework | Complex implementation | Multi-task scenarios |

### Data Processing Pipeline

| Stage | Purpose | Techniques | Challenges | Quality Impact |
|-------|---------|------------|------------|----------------|
| **Extraction** | HTML to text | Boilerplate removal, structure preservation | Format diversity | High |
| **Filtering** | Remove unwanted content | Language detection, quality classifiers | Balancing precision/recall | Very High |
| **Deduplication** | Remove duplicates | MinHash, exact matching | Near-duplicates, over-deduplication | Medium |
| **Mixing** | Combine data sources | Domain-specific sampling | Optimal ratios | High |

### Scaling Laws

**Key Finding**: Language modeling loss **predictably improves with more compute**

**Compute Formula**: `C ≈ 6ND` where N = parameters, D = tokens

**Scaling Law**: `L(C) ∝ 1/C^α` (e.g., α = 0.05)

**Applications**:
- Choose optimal model size for compute budget
- Estimate hyperparameters for large-scale training

---

## Inference and Generation

### Decoding Methods Comparison

| Method | Type | How It Works | Pros | Cons | Best For |
|--------|------|--------------|------|------|----------|
| **Greedy** | Deterministic | Pick highest probability token | Fast, deterministic | Can miss better sequences | Simple tasks |
| **Beam Search** | Deterministic | Keep top-k sequences | Better than greedy | Length bias, repetition, slow | Translation, summarization |
| **Top-k Sampling** | Stochastic | Sample from top-k tokens | Good diversity | Arbitrary cutoff | Creative tasks |
| **Top-p (Nucleus)** | Stochastic | Sample from cumulative p mass | Adaptive vocabulary | Complex tuning | General generation |
| **Temperature** | Modification | Scale logits before softmax | Simple control | Affects all tokens equally | Fine-tuning randomness |

### Decoding Issues and Solutions

| Problem | Symptoms | Solutions | Trade-offs |
|---------|----------|-----------|------------|
| **Length Bias** | Shorter outputs preferred | Length penalty, length normalization | May encourage verbosity |
| **Repetition** | Stuck in loops | Repetition penalty, n-gram blocking | May hurt natural repetition |
| **Atypicality** | Boring, generic outputs | Sampling methods, diverse beam search | Less coherent outputs |
| **Long Tail** | Incoherent rare words | Top-k, top-p filtering | May miss valid rare words |

### Constrained Generation Methods

| Method | Type | Flexibility | Speed | Implementation Complexity |
|--------|------|-------------|-------|-------------------------|
| **Templatic** | Hard constraints | Low | Fast | Medium |
| **Logit Manipulation** | Hard constraints | Medium | Fast | Low |
| **Sample-then-Rank** | Soft constraints | High | Slow | Low |
| **FUDGE** | Soft constraints | High | Medium | High |
| **RLHF** | Learned constraints | Very High | Medium | Very High |

---

## Fine-tuning and Adaptation

### Fine-tuning Methods Comparison

| Method | Data Format | Objective | Strengths | Limitations | Computational Cost |
|--------|-------------|-----------|-----------|-------------|-------------------|
| **Standard FT** | (input, output) pairs | Cross-entropy loss | Simple, effective | Task-specific, can overfit | High (full model) |
| **Instruction Tuning** | (instruction, input, output) | Multi-task cross-entropy | Generalizes to new tasks | Requires diverse instructions | High |
| **Chat Tuning** | Conversation format | Multi-turn dialogue loss | Natural conversations | Format-dependent | High |
| **LoRA** | Any task format | Low-rank parameter updates | Memory efficient | May miss some adaptations | Low |
| **QLoRA** | Any task format | Quantized + low-rank | Very memory efficient | Quantization artifacts | Very Low |

### Knowledge Distillation Approaches

| Type | Teacher Output | Student Learns | Advantages | Disadvantages | Use Case |
|------|----------------|----------------|------------|---------------|----------|
| **Token-level** | Probability distributions | To mimic token probabilities | Fine-grained learning | Requires teacher inference | Model compression |
| **Sequence-level** | Complete sequences | From generated outputs | Simpler, more flexible | Less detailed signal | Data augmentation |
| **Feature-level** | Intermediate representations | Internal representations | Rich signal | Requires architecture alignment | Model understanding |
| **Response-level** | Final answers only | Task performance | Task-focused | Coarse signal | Performance matching |

### Efficient Fine-tuning Comparison

| Method | Parameters Updated | Memory Usage | Performance | Flexibility | Best For |
|--------|-------------------|--------------|-------------|-------------|----------|
| **Full Fine-tuning** | All parameters | Very High (1000-1400GB for 65B) | Best | Full adaptation | When resources available |
| **LoRA** | Low-rank matrices | Medium | Good (90-95% of full) | Task-specific | Most applications |
| **QLoRA** | Quantized + low-rank | Low (48GB for 65B) | Good | Task-specific | Resource-constrained |
| **Prefix Tuning** | Prefix embeddings | Low | Moderate | Limited | Simple tasks |
| **Prompt Tuning** | Soft prompts | Very Low | Moderate | Very limited | Few-shot scenarios |

---

## Reinforcement Learning

### MLE vs RL Training Comparison

| Aspect | Maximum Likelihood (MLE) | Reinforcement Learning (RL) |
|--------|---------------------------|------------------------------|
| **Objective** | Maximize probability of training data | Maximize task-specific reward |
| **Data Source** | Fixed training dataset | Model-generated sequences |
| **Training Signal** | Teacher forcing on correct tokens | Reward on complete sequences |
| **Exposure Bias** | Never sees own mistakes | Trained on own generations |
| **Task Alignment** | Indirect (probability ≠ performance) | Direct (optimize actual metric) |
| **Stability** | Very stable | Can be unstable, needs regularization |
| **Computational Cost** | Lower | Higher (generation + reward computation) |
| **Data Requirements** | Needs paired (input, output) | Needs reward function |

### Reward Function Types

| Type | Examples | Advantages | Disadvantages | Implementation |
|------|----------|------------|---------------|----------------|
| **Rule-based** | Exact match, test pass rate | Objective, fast | Limited scope, brittle | Simple functions |
| **Model-based** | Preference models, quality classifiers | Flexible, nuanced | Can be biased, slower | Neural networks |
| **Human Feedback** | Direct ratings, preferences | High quality, aligned | Expensive, subjective | Crowdsourcing |
| **Hybrid** | Rule + model combination | Robust, flexible | Complex design | Multiple components |

### RL Stabilization Techniques

| Technique | Purpose | How It Works | Trade-offs | Critical For |
|-----------|---------|--------------|------------|--------------|
| **KL Penalty** | Prevent drift from reference | Add `βD_KL(π_θ || π_ref)` to reward | May limit exploration | Preventing reward hacking |
| **Baselines** | Reduce variance | Subtract expected reward | Requires baseline estimation | Stable learning |
| **PPO Clipping** | Prevent large updates | Clip policy ratio | May slow learning | Training stability |
| **Value Function** | Better advantage estimation | Learn state values | Additional complexity | Sample efficiency |

### RLHF Pipeline

1. **Supervised Fine-tuning (SFT)**: Fine-tune on instruction data
2. **Reward Modeling**: Train reward model on preference data
3. **Reinforcement Learning**: Fine-tune SFT model with RL using reward model

---

## Evaluation

### Properties of Good Benchmarks

1. **Difficulty**: Distinguish capable from less capable models
2. **Diversity**: Cover wide range of queries
3. **Usefulness**: High scores have practical meaning
4. **Reproducibility**: Consistent scores across runs
5. **Data Contamination**: Evaluate generalization, not memorization

### Evaluation Metrics Comparison

| Task Type | Metric | What It Measures | Pros | Cons | Best For |
|-----------|--------|------------------|------|------|----------|
| **Classification** | Accuracy | Exact match percentage | Simple, interpretable | No error type info | Balanced datasets |
| **Generation** | ROUGE | Word overlap | Fast, established | Misses semantic similarity | Summarization |
| **Generation** | BLEU | N-gram overlap | Standard for translation | Harsh on valid variations | Machine translation |
| **Generation** | BERTScore | Embedding similarity | Semantic awareness | Model-dependent | Semantic tasks |
| **Generation** | Exact Match | Perfect string match | Unambiguous | Too strict for creative tasks | Math, factual QA |
| **Code** | Pass@k | Test case success | Functional correctness | Limited test coverage | Programming tasks |
| **Dialogue** | Human Eval | Human preference | High quality | Expensive, subjective | Chat systems |

### Benchmark Categories

| Category | Examples | Evaluation Style | Strengths | Limitations | Current Status |
|----------|----------|------------------|-----------|-------------|----------------|
| **Reading Comprehension** | SQuAD, QuAC | Exact match, F1 | Clear metrics | Dataset artifacts | Some saturated |
| **Commonsense** | HellaSwag, PIQA | Multiple choice | Tests reasoning | Limited scope | Active research |
| **Knowledge** | MMLU, TriviaQA | Multiple choice/generation | Broad coverage | Memorization issues | Widely used |
| **Math** | GSM8K, MATH | Exact match | Verifiable | Limited problem types | Very active |
| **Code** | HumanEval, MBPP | Execution success | Functional testing | Test quality varies | Standard benchmark |
| **Multi-step** | DROP, StrategyQA | Complex reasoning | Tests capabilities | Hard to scale | Emerging focus |

---

## Advanced Topics

### Long-Context Models

**Challenges**:
- **Memory**: Quadratic scaling `O(s²)`
- **Compute**: Quadratic complexity
- **Lost-in-the-middle**: Poor attention to middle content

**Solutions**:
- **Efficient Attention**: Linear attention, Flash Attention
- **Extrapolation**: Train on longer sequences, RoPE scaling
- **State Space Models**: Mamba, S4 with linear scaling

### Inference Scaling

**Parallel Strategies**:
- **Best-of-N**: Select best from N candidates using reward model
- **Voting**: Majority vote on final answers
- **MBR**: Select candidate with highest consensus utility

**Sequential Strategies**:
- **Tree Search**: Beam search, MCTS on reasoning paths
- **Long Chain-of-Thought**: Generate extended reasoning before answer

### Multimodal Models

**Vision-Language Models**:
- **CLIP**: Contrastive learning on image-text pairs
- **VLMs**: Combine vision encoder with language model
- **Vision Transformer**: Process images as sequence of patches

**Image Generation**:
- **VQ-VAE**: Discrete image tokens for autoregressive modeling
- **Diffusion Models**: Gradual denoising process

### Efficient Training and Inference

#### Parallelization Strategies

| Strategy | How It Works | Memory Distribution | Communication | Best For | Limitations |
|----------|--------------|-------------------|---------------|----------|-------------|
| **Data Parallel** | Split batch across GPUs | Model replicated on each GPU | Gradient synchronization | Large batches, smaller models | Memory per GPU = full model |
| **Tensor Parallel** | Split layers across GPUs | Model weights distributed | High (activations) | Large models, fast interconnect | Communication overhead |
| **Pipeline Parallel** | Split model depth across GPUs | Sequential layer distribution | Medium (between stages) | Very large models | Pipeline bubbles, load balancing |
| **3D Parallel** | Combine all three | Hybrid distribution | Optimized patterns | Massive models | Complex orchestration |

#### Quantization Methods

| Method | Precision | Techniques | Memory Reduction | Performance Impact | Use Case |
|--------|-----------|------------|------------------|-------------------|----------|
| **FP16/BF16** | 16-bit | Half precision | 50% | Minimal | Standard training |
| **INT8** | 8-bit | Post-training quantization | 75% | Small | Inference optimization |
| **INT4** | 4-bit | Advanced techniques (NF4) | 87.5% | Moderate | Memory-constrained inference |
| **QLoRA** | 4-bit + adapters | Quantization + LoRA | 90%+ | Maintained via fine-tuning | Efficient fine-tuning |
| **1-bit** | Binary | Extreme quantization | 96.875% | Significant | Research/edge devices |

#### Inference Optimization Techniques

| Technique | Method | Latency Impact | Throughput Impact | Memory Impact | Complexity |
|-----------|--------|----------------|-------------------|---------------|------------|
| **Speculative Decoding** | Draft model + verification | Reduces | Increases | Slight increase | Medium |
| **KV Cache Optimization** | Efficient storage/retrieval | Reduces | Increases | Optimizes usage | Low |
| **Dynamic Batching** | Variable batch sizes | Variable | Maximizes | Efficient usage | Medium |
| **Model Sharding** | Distribute across devices | Depends on setup | Increases | Distributes load | High |
| **Quantized Inference** | Lower precision | Reduces | Increases | Significantly reduces | Low-Medium |

---

## Interview Tips

### Key Concepts to Master

1. **Transformer Architecture**: Understand every component deeply
2. **Attention Mechanism**: Different types and when to use them
3. **Training Objectives**: MLE vs RL, their trade-offs
4. **Scaling Laws**: Relationship between compute, data, and performance
5. **Evaluation**: How to properly evaluate LLMs

### Common Question Patterns

1. **Architecture Questions**: "Explain how attention works"
2. **Training Questions**: "Why use RLHF instead of just supervised fine-tuning?"
3. **Practical Questions**: "How would you reduce inference latency?"
4. **Trade-off Questions**: "What are the pros/cons of different decoding methods?"
5. **Recent Advances**: "What are the latest developments in efficient training?"

### Deep Dive Areas

- Understand mathematical formulations
- Know implementation details and practical considerations
- Be familiar with recent papers and their contributions
- Understand system-level considerations (memory, compute, latency)
- Know evaluation methodologies and their limitations

---

*This guide covers the essential theoretical foundations needed for LLM interviews. Focus on understanding the underlying principles and their practical implications.*