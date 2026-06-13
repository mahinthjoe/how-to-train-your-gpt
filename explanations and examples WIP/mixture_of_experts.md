# Mixture of Experts: How to Scale Without Scaling Everything

## The short answer

A Mixture of Experts model has many feed forward networks instead of
one. Each token is routed to only a few of them. The model has far
more total parameters than a standard model but uses only a fraction
of them for any given token. This is how models like Mixtral 8x7B
and GPT-4 achieve the performance of much larger models while keeping
the compute cost of much smaller ones.

Think of it like a hospital. A standard model has one doctor who
treats every patient. An MoE model has many specialist doctors. Each
patient is sent to the two or three doctors most qualified to handle
their case. The hospital has more total expertise but each patient
only spends time with the relevant specialists.

## Where it sits

MoE replaces the feed forward network in each transformer block.
Attention stays the same. Normalization stays the same. Only the
FFN changes from a single dense network to a collection of experts
with a router that decides which experts to use.

```
Standard transformer block:
  x → RMSNorm → Attention → +x
    → RMSNorm → FFN → +x

MoE transformer block:
  x → RMSNorm → Attention → +x
    → RMSNorm → Router → Expert 1 (used 30% of the time)
                      → Expert 2 (used 25%)
                      → Expert 3 (used 20%)
                      → ...
                      → Expert 8 (used 5%)
                                → Combine outputs → +x
```

Each expert is an entire feed forward network. Same architecture as
our standard FFN with SwiGLU. Same input and output dimensions. The
difference is that there are eight of them instead of one.

## Why MoE works

Dense models use every parameter for every token. This is wasteful.
Some parameters handle grammar. Some handle facts. Some handle
reasoning. For any given token only a subset of parameters is
actually useful. The rest contribute noise or are effectively dead.

MoE lets different tokens use different paths through the model. A
token representing a verb might use experts that specialize in verb
conjugation and argument structure. A token representing a noun
might use experts that specialize in entity recognition and coreference.
The router learns to send each token to the right experts.

The total parameter count is larger because experts are duplicated.
But the compute cost per token is roughly the same because only a
few experts are activated. The model gets more capacity without more
compute. This is capacity with low active parameters.

## The router

The router is a small linear layer. It takes the hidden state of a
token and outputs a score for each expert. The token is sent to the
experts with the highest scores.

```python
class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        # x: [batch, seq, d_model]
        logits = self.gate(x)  # [batch, seq, num_experts]

        # Select top-k experts per token
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_indices, top_k_weights
```

For each token the router selects the top two experts. It also
outputs weights for each selected expert. The weights determine
how much each expert contributes to the final output. The weights
are computed by applying softmax to the top-k logits so they sum
to one.

## Top-k selection

The model does not use all experts. It uses only the top two per
token. This is the key to MoE efficiency. With eight experts and
top-2 selection only 25 percent of the FFN parameters are active
for each token. The model has 8x the FFN parameters of a dense
model but only 2x the compute.

```
Dense model FFN:    768 → 3072 → 768   (7.1M params per block)
MoE model FFN:      8 × (768 → 3072 → 768) = 56.7M params per block
Active per token:   2 × (768 → 3072 → 768) = 14.2M params used

Total FFN params:   8× larger
Active compute:     2× larger (top-2 of 8)
```

You get 8x the capacity for 2x the compute. This ratio improves as
the number of experts grows. With 64 experts and top-2 selection you
get 64× capacity for 2× compute.

## Load balancing

The router can learn to always send tokens to the same expert. Expert
1 gets 80 percent of the tokens. Experts 2 through 8 are idle. The
model becomes effectively a dense model with seven wasted experts.

Preventing this requires a load balancing loss. The loss encourages
the router to distribute tokens evenly across experts. Each expert
should receive roughly the same number of tokens over the course of
training.

```python
def load_balancing_loss(router_logits, expert_mask, num_experts):
    """
    router_logits: [batch * seq, num_experts] : raw router scores
    expert_mask:   [batch * seq, num_experts] : 1 if expert was selected

    Returns a scalar loss that penalizes uneven routing.
    The loss is zero when every expert receives equal tokens.
    """
    # Fraction of tokens routed to each expert
    fraction_per_expert = expert_mask.float().mean(dim=0)

    # Average router probability for each expert
    router_probs = F.softmax(router_logits, dim=-1)
    avg_prob_per_expert = router_probs.mean(dim=0)

    # Loss = num_experts * sum(fraction × probability)
    # Minimized when fractions and probabilities are uniform
    return num_experts * (fraction_per_expert * avg_prob_per_expert).sum()
```

The load balancing loss is added to the main language modeling loss
with a small coefficient typically 0.01. It pushes the router toward
uniform usage without dominating the training objective.

## Expert capacity

Even with load balancing some experts may receive more tokens than
others in a given batch. To prevent memory spikes each expert has a
capacity limit. If more tokens are routed to an expert than its
capacity the overflow tokens are dropped. They pass through the
residual connection unchanged.

```
Expert capacity = (tokens_per_batch / num_experts) × capacity_factor

capacity_factor = 1.25 means each expert can handle 25% more tokens
than its fair share. Overflow tokens above this limit are dropped.
```

The capacity factor is typically 1.0 to 1.5. Higher values waste
compute because experts are allocated for tokens they never receive.
Lower values increase the chance of dropping tokens which degrades
model quality.

Dropped tokens are not a complete loss. The residual connection
still carries the token's information forward. The token skips the
FFN for this layer and gets processed by the next layer's experts.

## The full MoE transformer block

```python
class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_experts=8, top_k=2):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)

        # MoE replaces the single FFN
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(d_model) for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x, mask=None):
        batch, seq, d_model = x.shape

        # Attention
        x = x + self.attention(self.norm1(x), mask)

        # MoE FFN
        residual = x
        x = self.norm2(x)

        # Route tokens to experts
        logits = self.router(x)  # [batch, seq, num_experts]
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [batch, seq, top_k]

        # Process tokens through selected experts
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            # Get the tokens for this expert
            tokens = x[expert_mask]  # [num_routed, d_model]

            # Run through the expert
            expert_output = self.experts[expert_idx](tokens)

            # Weight by the router weight for this expert
            weight_idx = (top_k_indices == expert_idx).float().argmax(dim=-1)
            weights = top_k_weights[expert_mask]
            weights = weights.gather(-1, weight_idx.unsqueeze(-1))

            # Accumulate weighted output
            output[expert_mask] += weights * expert_output

        x = residual + output
        return x
```

The forward pass loops over experts. This is slow in pure Python
but fast in optimized implementations that batch all expert
computations into a single matrix multiplication. Libraries like
Megablocks and DeepSpeed MoE handle this efficiently.

## Which models use MoE

| Model | Total Params | Active Params | Experts per Layer | Top-k |
|---|---|---|---|---|
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 |
| Mixtral 8x22B | 141B | 39B | 8 | 2 |
| GPT-4 (rumored) | ~1.7T | ~280B | 8 or 16 | 2 |
| Gemini 1.5 (rumored) | Unknown | Unknown | Unknown | Unknown |
| Switch Transformer | 1.6T | 1.6T | 2048 | 1 |
| GLaM | 1.2T | 96B | 64 | 2 |

The active parameters column is what matters for inference speed.
Mixtral 8x7B has 46.7 billion parameters on disk but only 12.9
billion are active per token. It runs roughly as fast as a 13
billion parameter dense model while matching the quality of a much
larger one.

Switch Transformer used top-1 routing which sends each token to
exactly one expert. This maximizes efficiency but hurts quality
because tokens cannot benefit from multiple specialists. Most
modern MoE models use top-2 as the sweet spot.

## Why not use MoE everywhere

MoE has downsides. The model is physically larger requiring more
memory to store. Inference memory is dominated by the KV cache not
the model weights for long sequences so this matters less than it
seems. But loading a 47 billion parameter model still requires more
VRAM than loading a 13 billion parameter one even if they run at
similar speeds.

Training is harder. The load balancing loss and expert capacity
constraints add complexity. The router can collapse to always using
the same expert which requires monitoring and intervention. MoE
models are more prone to training instability than dense models.

Fine-tuning MoE models is trickier. With LoRA you need to decide
whether to adapt the experts or the router or both. The interactions
between experts and router are not yet as well understood as standard
transformer components.

## Should we add MoE to our model

Our model is too small to benefit from MoE. The expert count should
exceed the number of useful specializations. With only 4 layers and
152 million parameters there are not enough distinct linguistic
patterns for experts to specialize in. A single FFN per layer is
sufficient.

MoE starts to help around the 1 billion parameter mark. Below that
the overhead of routing and load balancing outweighs the benefit of
additional capacity. The sweet spot for MoE is models with 10 billion
or more parameters where the compute savings from sparse activation
are substantial.

## What you need to remember

A Mixture of Experts model replaces each feed forward network with
multiple expert networks. A router sends each token to the top one
or two experts. The model has more total parameters but uses only a
fraction per token. This gives the capacity of a large model at the
compute cost of a smaller one.

Load balancing ensures all experts get used rather than collapsing
to a single favored expert. Expert capacity limits prevent memory
spikes from uneven routing. The router and experts are trained
jointly with the rest of the model.

MoE is the likely architecture behind GPT-4 and Gemini. It is the
most practical way to scale models beyond the point where dense
training becomes prohibitively expensive. Understanding MoE is
understanding how the largest AI systems are built.
