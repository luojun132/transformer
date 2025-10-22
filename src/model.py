import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear projection
        output = self.W_o(attn_output)

        return output, attn_weights


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualConnection(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, mask)[0])
        x = self.residual2(x, self.feed_forward)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)

        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self attention (with look-ahead mask)
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, tgt_mask)[0])
        # Cross attention with encoder output
        x = self.residual2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)[0])
        # Feed forward
        x = self.residual3(x, self.feed_forward)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_len, dropout=0.1, use_positional_encoding=True):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.use_positional_encoding = use_positional_encoding

        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.token_embedding(x)

        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_len, dropout=0.1, use_positional_encoding=True):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.use_positional_encoding = use_positional_encoding

        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.token_embedding(x)

        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


# ==================== 模型定义 ====================

class TransformerLM(nn.Module):
    """Transformer Language Model (Encoder-only)"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_len, dropout=0.1, use_positional_encoding=True):
        super(TransformerLM, self).__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout, use_positional_encoding
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        encoder_output = self.encoder(x, mask)
        logits = self.lm_head(encoder_output)
        return logits


class TransformerSeq2Seq(nn.Module):
    """完整的Transformer (Encoder-Decoder) 用于序列到序列任务"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_seq_len, dropout=0.1, use_positional_encoding=True):
        super(TransformerSeq2Seq, self).__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout, use_positional_encoding
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout, use_positional_encoding
        )

        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_layer(decoder_output)
        return output

    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_layer(decoder_output)