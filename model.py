import math
import mlx.core as mx
import mlx.nn as nn

class RobertaConfig:
    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=6, 
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=514,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        **kwargs # Allow extra keys
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    @classmethod
    def from_dict(cls, config_dict):
        # Force layer_norm_eps to float
        if 'layer_norm_eps' in config_dict:
             config_dict['layer_norm_eps'] = float(config_dict['layer_norm_eps'])
        return cls(**config_dict)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def __call__(self, x, mask=None):
        B, L, D = x.shape
        queries = self.query(x).reshape(B, L, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.key(x).reshape(B, L, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.value(x).reshape(B, L, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled Dot Product
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        # attn_weights = self.dropout(attn_weights) # Dropout usually before matmul? or on weights? usually on weights
        
        out = (attn_weights @ values).transpose(0, 2, 1, 3).reshape(B, L, D)
        out = self.out_proj(out)
        return out

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, x, mask=None):
        attention_out = self.attention(self.ln1(x), mask)
        x = x + attention_out # Residual
        
        mlp_out = self.linear2(self.dropout(self.gelu(self.linear1(self.ln2(x)))))
        x = x + mlp_out # Residual
        return x

class RobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Simplified embeddings: Just lookup
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # Position embeddings (learnable)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.ln_emb = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = [TransformerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, input_ids, mask=None):
        B, L = input_ids.shape
        
        # Embeddings
        w_emb = self.word_embeddings(input_ids)
        
        # Positions: 0 to L-1
        # In RoBERTa there's usually a padding index offset, but for simple pretraining 0-based is fine if consistent
        pos_ids = mx.arange(L)[None]
        p_emb = self.position_embeddings(pos_ids)
        
        x = w_emb + p_emb
        x = self.ln_emb(x)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

class RobertaForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        # Usually there is an extra dense+LN+Gelu layer in the head, but simple linear works for simple pipelines

    def __call__(self, input_ids, mask=None):
        x = self.roberta(input_ids, mask)
        logits = self.lm_head(x)
        return logits
