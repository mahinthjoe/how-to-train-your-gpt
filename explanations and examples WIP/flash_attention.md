# Flash Attention: Making Attention Fast

## The short answer

Flash Attention is not a new type of attention. It is the same math
running faster. Standard attention computes the full Q times K
transpose matrix then applies softmax then multiplies by V. This
requires reading and writing a massive intermediate matrix from GPU
memory. Flash Attention avoids storing that matrix entirely. It
computes attention in small blocks and keeps everything inside the
GPU's fast on-chip memory. The result is two to four times faster
training and generation for long sequences.

Every major LLM training run since 2022 has used Flash Attention.
It is not an optional optimization. It is the difference between
training on 2048 tokens and training on 32768 tokens on the same GPU.

## Where the bottleneck is

GPUs have two kinds of memory. The big memory called HBM or VRAM
stores everything. Model weights. KV caches. Training data. It is
large but slow. Reading from it takes hundreds of cycles. The small
memory called SRAM lives on the GPU chip itself. It is tiny but
extremely fast. A few megabytes. Reading from it takes a few cycles.

The standard attention implementation writes the full attention matrix
to HBM. Then reads it back for softmax. Then writes the weights back
to HBM. Then reads them back for the V multiplication. Each of these
reads and writes is a bottleneck.

```
Standard attention memory flow:

  Q, K, V in HBM
    → Load Q, K to compute scores
    → Write scores matrix to HBM  (seq_len × seq_len)
    → Read scores from HBM for softmax
    → Write weights matrix to HBM (seq_len × seq_len)
    → Read weights and V from HBM for output computation
    → Write output to HBM

  Problem: The seq_len × seq_len matrix is huge.
  For 8192 tokens: 8192 × 8192 × 2 bytes = 134 MB just for scores.
  Every transformer layer reads and writes this.
  96 layers × 134 MB = ~13 GB of HBM traffic just for attention.
```

Flash Attention never writes the full matrix. It processes attention
in small tiles that fit entirely in SRAM. The compute happens on-chip.
The output is the only thing written back to HBM.

```
Flash Attention memory flow:

  Q, K, V in HBM
    → Load a tile of Q into SRAM
    → Load a tile of K into SRAM
    → Compute scores for this tile pair
    → Apply softmax online (incrementally)
    → Load corresponding V tile
    → Accumulate weighted values in SRAM
    → Write only the final output tile to HBM

  Result: The seq_len × seq_len matrix never exists in HBM.
  Memory traffic is proportional to seq_len, not seq_len squared.
```

## The two key ideas

### Tiling

Instead of processing the entire Q and K matrices at once split them
into small tiles. A tile of Q is 128 tokens by 64 dimensions. A tile
of K is 128 tokens by 64 dimensions. The attention scores between
these tiles is 128 by 128. That fits in SRAM. Process one pair of
tiles at a time. Accumulate results.

This is like reading a book one page at a time instead of spreading
all the pages on the floor. You can only hold one page in your hands
at a time but you can read the whole book without needing a bigger
table.

### Online softmax

Softmax is usually computed in two passes. First find the maximum
value for numerical stability. Then compute exponentials and divide
by the sum. This requires having all scores available at once.

Online softmax computes softmax incrementally as each tile of scores
arrives. It maintains a running maximum and running sum. When a new
tile arrives with a larger maximum it rescales the previous results.
When the last tile is processed the final output is ready. No second
pass needed.

```
Online softmax for tile i:

  m_new = max(m_old, max(scores_i))
  sum_new = sum_old × exp(m_old - m_new) + sum(exp(scores_i - m_new))
  output = (output_old × sum_old × exp(m_old - m_new)
           + V_i × sum(exp(scores_i - m_new))) / sum_new
  m_old = m_new
  sum_old = sum_new
```

The rescaling by `exp(m_old - m_new)` corrects the old output when the
maximum increases. This is mathematically identical to computing
softmax on all scores at once. Just done in pieces.

## Why it matters for the causal mask

Flash Attention with a causal mask is even faster. Tiles in the upper
triangle of the attention matrix are zero. There is no need to compute
them at all. Skip them entirely. For a sequence of length N only about
half the tiles need computation. The other half are known to be zero
from the mask alone.

```
Causal attention tile pattern (N=6 tiles of 128 tokens each):

            K0  K1  K2  K3  K4  K5
         Q0 ██  ░░  ░░  ░░  ░░  ░░
         Q1 ██  ██  ░░  ░░  ░░  ░░
         Q2 ██  ██  ██  ░░  ░░  ░░
         Q3 ██  ██  ██  ██  ░░  ░░
         Q4 ██  ██  ██  ██  ██  ░░
         Q5 ██  ██  ██  ██  ██  ██

  ██ = compute this tile
  ░░ = skip (all zeros from causal mask)

  Computed tiles: 21 of 36 = 58 percent of the full matrix
```

Standard attention still writes zeros for the upper triangle tiles.
Flash Attention never touches them. It computes only the lower
triangle tiles. For long sequences this roughly halves the already
reduced workload.

## The speed difference

For training with sequence length 2048 on an A100 GPU.

```
Standard attention:     85 ms per forward pass
Flash Attention V1:     24 ms per forward pass  (3.5x faster)
Flash Attention V2:     19 ms per forward pass  (4.5x faster)

Memory used for attention scores:
Standard attention:     64 MB per layer
Flash Attention:         0 MB per layer (never materialized)

Maximum sequence length on 40 GB A100:
Standard attention:     ~4096 tokens
Flash Attention:        ~16384 tokens (4× longer)
```

The speed difference grows with sequence length. At 8192 tokens
Flash Attention can be 8x faster than standard attention. The memory
savings mean you can train on sequences four times longer or use four
times the batch size on the same GPU.

## A simplified code sketch

This is not production code. It shows the structure. Real Flash
Attention is written in CUDA C++ and highly optimized for specific
GPU architectures.

```python
def flash_attention_forward(Q, K, V, causal=True, block_size=128):
    """
    Simplified flash attention. Processes attention in blocks
    to avoid materializing the full N×N attention matrix.
    """
    batch, heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)

    output = torch.zeros_like(Q)

    for i in range(0, seq_len, block_size):
        q_block = Q[:, :, i:i + block_size]
        q_max = torch.full((batch, heads, block_size, 1),
                           float('-inf'), device=Q.device)
        q_sum = torch.zeros(batch, heads, block_size, 1, device=Q.device)
        out_block = torch.zeros_like(q_block)

        j_end = (i + block_size) if causal else seq_len
        for j in range(0, j_end, block_size):
            k_block = K[:, :, j:j + block_size]
            v_block = V[:, :, j:j + block_size]

            scores = (q_block @ k_block.transpose(-2, -1)) * scale

            new_max = torch.maximum(q_max, scores.max(dim=-1, keepdim=True).values)
            correction = torch.exp(q_max - new_max)

            exp_scores = torch.exp(scores - new_max)
            new_sum = correction * q_sum + exp_scores.sum(dim=-1, keepdim=True)

            out_block = (correction * q_sum / new_sum) * out_block
            out_block = out_block + (exp_scores / new_sum) @ v_block

            q_max = new_max
            q_sum = new_sum

        output[:, :, i:i + block_size] = out_block

    return output
```

The crucial detail: `out_block` is accumulated in place across the
inner loop without ever storing the full scores matrix. The running
max and sum track the softmax state. When the maximum increases the
correction term rescales the previous output. When all K tiles are
processed the block of Q tiles has its complete attention output.

## Can I use Flash Attention in this project

Yes. The pip package is called `flash-attn`. Install it and replace
the attention forward pass with `flash_attn_func`. But there are two
catches.

First it only works on NVIDIA GPUs with compute capability 8.0 or
higher. That means A100 H100 RTX 3090 RTX 4090. It does not work on
CPU or Apple MPS or older GPUs.

Second the API expects a slightly different tensor layout. The input
must be in shape batch seq_len num_heads head_dim with the sequence
dimension before the head dimension. Our code uses batch num_heads
seq_len head_dim. You need to transpose.

```python
# Using Flash Attention with our model
from flash_attn import flash_attn_func

# Our attention (simplified):
qkv = self.qkv_proj(x)
qkv = qkv.reshape(batch, seq, 3, num_heads, head_dim)
qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
q, k, v = qkv[0], qkv[1], qkv[2]

# Flash Attention expects [batch, seq, heads, head_dim]:
q = q.permute(0, 2, 1, 3)  # [batch, seq, heads, head_dim]
k = k.permute(0, 2, 1, 3)
v = v.permute(0, 2, 1, 3)

output = flash_attn_func(q, k, v, causal=True)

output = output.permute(0, 2, 1, 3)  # Back to [batch, heads, seq, head_dim]

# Continue as normal
output = output.transpose(1, 2).contiguous()
output = output.reshape(batch, seq, d_model)
output = self.out_proj(output)
```

## What you need to remember

Flash Attention is the same attention math running faster. It never
writes the full N by N attention matrix to GPU memory. Instead it
processes attention in small tiles using fast on-chip memory. Online
softmax computes the softmax incrementally without a second pass.

For sequences under 512 tokens the difference is small. For sequences
over 2048 tokens Flash Attention is 3x to 8x faster. It also reduces
memory usage by not storing the attention matrix enabling longer
sequences or larger batch sizes on the same GPU.

Every production LLM training run since 2022 uses Flash Attention.
It is not an optional optimization for serious work. It is a
requirement.
