import numpy as np
import tensorflow as tf
import torch.nn as nn

'''
Multi-headed attention
'''
# Input sequence: 10 words, each represented by a 3-dimensional vector
sequence_length, dimension, batch_size = 10, 3, 2
input_sequence = tf.random.normal((batch_size, sequence_length, dimension))

# Multi-head attention layer with 2 attention heads
num_attention_heads = 2
multi_head_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_attention_heads, key_dim=dimension)

# Self-attention: query, key, and value are all derived from the input sequence
output_sequence = multi_head_layer(query=input_sequence, value=input_sequence, key=input_sequence)

print(output_sequence.shape)  # Output: (2, 10, 3)


'''
Positional Encoding
'''
def positional_encoding(seq_length, d_model):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# Example usage
seq_length, d_model = 100, 512
positional_encodings = positional_encoding(seq_length, d_model)

'''
BERT
'''
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8),
            num_layers=num_layers
        )
    
    def forward(self, x):
        x = self.embedding(x)
        return self.transformer(x)

'''
GPT
'''

class GPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, nhead=8),
            num_layers=num_layers
        )
    
    def forward(self, x):
        x = self.embedding(x)
        return self.transformer(x, x)