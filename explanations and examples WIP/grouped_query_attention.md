# Grouped Query Attention and Multi-Query Attention

## The short answer

Standard multi-head attention gives every attention head its own Query
Key and Value. Grouped Query Attention shares Key and Value heads
across groups of Query heads. One Key and Value pair serves multiple
Queries. This cuts the KV cache size dramatically during inference.
The model quality barely drops. Multi-Query Attention takes this to
the extreme by using a single Key and Value head for all Query heads.

## Where this matters

Every time the model generates a new token during inference it must
store the Key and Value vectors for that token in the KV cache. With
standard multi-head attention the cache stores one K and one V per
head per layer per token.

```
Standard MHA (12 heads):
  Each new token adds: 12 K vectors + 12 V vectors
  Cache size for 1000 tokens: 24 × 1000 × head_dim floats

GQA (12 Q heads, 4 KV heads):
  Each new token adds: 4 K vectors + 4 V vectors
  Cache size for 1000 tokens: 8 × 1000 × head_dim floats
  Memory savings: 3× smaller cache

MQA (12 Q heads, 1 KV head):
  Each new token adds: 1 K vector + 1 V vector
  Cache size for 1000 tokens: 2 × 1000 × head_dim floats
  Memory savings: 12× smaller cache
```

The memory savings are real. For a 70 billion parameter model generating
4096 tokens the KV cache can be gigabytes. Cutting it by 4x or 8x means
the difference between fitting in GPU memory and crashing.

## Why this works

Do we really need twelve separate Key and Value heads. Each head in
standard MHA has its own perspective on the input. The Query heads
need this diversity. Different queries look for different patterns.
Grammar heads look for subject verb agreement. Semantic heads look for
meaning relationships. Position heads look for word order.

But the Keys and Values do not need as much diversity. They represent
what each token has to offer. A token offers the same fundamental
information regardless of which Query head is looking at it. The verb
*sat* offers the same semantic content whether the grammar head or the
semantic head is querying it. Sharing Keys and Values across heads
loses little information.

GQA strikes a balance. A few KV heads provide enough diversity for the
query heads to find what they need. The optimal ratio is about four to
eight query heads per KV head. Beyond that the quality drops noticeably.

## The code change from standard MHA

The difference is minimal. Standard MHA projects to 3 × d_model for
Q K and V. GQA projects to d_model × d_model + 2 × kv_heads × head_dim.

```python
# Standard Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Projects to: Q (d_model) + K (d_model) + V (d_model)
        # All heads get their own Q, K, V

# Grouped Query Attention
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads

        # Q projection: full size (1 per head)
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # K and V projections: smaller (1 per KV head)
        kv_dim = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
```

The key difference: `k_proj` and `v_proj` project to `num_kv_heads × head_dim`
instead of `num_heads × head_dim`. Fewer KV projections. Less memory.
Faster inference.

### The forward pass

The forward pass needs one extra step. The KV heads must be repeated to
match the number of query heads.

```python
def forward(self, x, mask=None):
    batch, seq, d_model = x.shape

    # Project Q fully, K and V with fewer heads
    q = self.q_proj(x)
    q = q.reshape(batch, seq, self.num_heads, self.head_dim)
    q = q.permute(0, 2, 1, 3)

    k = self.k_proj(x)
    k = k.reshape(batch, seq, self.num_kv_heads, self.head_dim)
    k = k.permute(0, 2, 1, 3)

    v = self.v_proj(x)
    v = v.reshape(batch, seq, self.num_kv_heads, self.head_dim)
    v = v.permute(0, 2, 1, 3)

    # Repeat KV heads to match query heads
    # Example: 12 Q heads, 4 KV heads → repeat each KV 3 times
    repeat_factor = self.num_heads // self.num_kv_heads
    k = k.repeat_interleave(repeat_factor, dim=1)
    v = v.repeat_interleave(repeat_factor, dim=1)

    # Standard attention from here
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    output = weights @ v

    output = output.transpose(1, 2).contiguous()
    output = output.reshape(batch, seq, d_model)
    return self.out_proj(output)
```

The `repeat_interleave` on line 25 is the only new operation. It takes
the four KV heads and repeats each one three times to get twelve. The
twelve query heads can then attend to these repeated KV heads.

During inference the KV cache stores only `num_kv_heads` Keys and Values
per layer. The repeat happens on the fly. The computation is the same
as standard attention. Only the memory footprint changes.

## Multi-Query Attention: the extreme case

MQA is GQA with `num_kv_heads` set to 1. One Key and one Value for all
query heads. The repeat factor equals `num_heads`.

```python
# MQA: num_kv_heads = 1
# k_proj projects to just 1 × head_dim dimensions
# v_proj projects to just 1 × head_dim dimensions
# repeat_interleave(num_heads) duplicates to match query heads
```

MQA was introduced by Google in 2019 for translation models. It saves
the maximum memory but quality drops more than GQA. PaLM and Gemini
use MQA successfully because their models are large enough that the
quality loss from sharing is offset by the quality gain from scale.

## Which models use which

| Model | Architecture | Q Heads | KV Heads | Ratio |
|---|---|---|---|---|
| GPT-2 | MHA | 12 | 12 | 1:1 |
| GPT-3 | MHA | 96 | 96 | 1:1 |
| LLaMA 1 | MHA | varies | varies | 1:1 |
| LLaMA 2 7B/13B | MHA | 32 | 32 | 1:1 |
| LLaMA 2 70B | GQA | 64 | 8 | 8:1 |
| LLaMA 3 8B | GQA | 32 | 8 | 4:1 |
| LLaMA 3 70B | GQA | 64 | 8 | 8:1 |
| Mistral 7B | GQA | 32 | 8 | 4:1 |
| Gemma 7B | MHA | 16 | 16 | 1:1 |
| PaLM | MQA | varies | 1 | varies |
| Gemini | MQA | varies | 1 | varies |

The trend is clear. Smaller models keep MHA because the KV cache is
small enough anyway. Larger models switch to GQA because the memory
savings matter at scale. A few very large models push to MQA for
maximum savings.

## When to use GQA in your own model

If your model is under about 13 billion parameters MHA is fine. The
KV cache is small. The memory savings from GQA do not justify the
engineering complexity.

If your model is between 13 billion and 70 billion parameters GQA is
the sweet spot. Use a ratio of 4:1 or 8:1. The memory savings are
significant. The quality loss is barely measurable.

If your model is over 70 billion parameters and you are serving many
concurrent users consider MQA. The extreme memory savings let you
serve more users per GPU. The quality loss is noticeable but the
economics often win.

## A complete GQA code example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.repeat = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch, seq, _ = x.shape

        q = self.q_proj(x)
        q = q.reshape(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        k = self.k_proj(x)
        k = k.reshape(batch, seq, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.repeat_interleave(self.repeat, dim=1)

        v = self.v_proj(x)
        v = v.reshape(batch, seq, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.repeat_interleave(self.repeat, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        output = weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch, seq, self.num_heads * self.head_dim)
        return self.out_proj(output)


# Memory comparison
d_model = 4096
num_heads = 32
num_kv_heads = 8
seq_len = 2048
head_dim = d_model // num_heads

mha_kv_size = 2 * num_heads * seq_len * head_dim          # K + V
gqa_kv_size = 2 * num_kv_heads * seq_len * head_dim        # K + V

print(f"MHA KV cache per layer: {mha_kv_size * 2 / 1e6:.1f} MB (bfloat16)")
print(f"GQA KV cache per layer: {gqa_kv_size * 2 / 1e6:.1f} MB (bfloat16)")
print(f"Memory savings: {mha_kv_size / gqa_kv_size:.1f}x")

# For 80 layers:
print(f"\nFor an 80 layer model:")
print(f"MHA total KV cache: {mha_kv_size * 80 * 2 / 1e9:.2f} GB")
print(f"GQA total KV cache: {gqa_kv_size * 80 * 2 / 1e9:.2f} GB")
```

## What you need to remember

Grouped Query Attention shares Key and Value heads across groups of
Query heads. This reduces the KV cache size during inference by the
ratio of query heads to KV heads. A ratio of 4:1 or 8:1 is standard.
The quality loss is minimal. The memory savings are dramatic at scale.

Multi-Query Attention is the extreme case with a single KV head for
all query heads. Maximum memory savings but noticeable quality loss.
Used by PaLM and Gemini where the model is large enough to compensate.

The code difference from standard attention is minimal. Separate Q K
and V projections with different output sizes. One repeat interleave
operation. The rest of the attention computation is identical.

For models under 13 billion parameters use standard MHA. For larger
models switch to GQA. The memory savings make the difference between
a model that serves users and a model that runs out of GPU memory
after a few hundred tokens.
