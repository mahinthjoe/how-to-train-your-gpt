# Perplexity: The One Number That Measures Your Model

## The short answer

Perplexity is a single number that tells you how good a language model
is. Lower is better. A perplexity of 10 means the model is as confused
as if it had to choose between 10 equally likely words at every step.
A perplexity of 1 means the model knows exactly what word comes next
every time. Real language models on real text usually have perplexity
between 10 and 100.

Perplexity is not abstract. It is the exponentiated cross entropy loss.
If your training loss is 3.0 your perplexity is e to the power of 3
which is about 20. The model is as confused as if picking randomly
among 20 options.

## Where the number comes from

Every time the model predicts the next word it assigns a probability to
every token in the vocabulary. The correct token gets some probability
P. The model is uncertain about that prediction. Cross entropy loss
measures that uncertainty as negative log of P. Perplexity is e to the
power of that loss.

```
For a single prediction:
  Model says P("mat") = 0.25
  Loss = -ln(0.25) = 1.386
  Perplexity = e^1.386 = 4.0

Interpretation: The model was as uncertain as if it had to pick
randomly among 4 equally likely options.
```

The magic of perplexity is that it translates an abstract loss number
into something you can visualize. A loss of 1.386 means nothing to most
people. A perplexity of 4 means the model is choosing among 4 options.
That is tangible. You can imagine picking randomly from 4 words.

## Perplexity versus loss

During training you watch the loss go down. The loss starts around 10.8
for a model with 50257 vocabulary tokens. That is uninterpretable. But
you can convert it.

```
loss = 10.82:  perplexity = e^10.82 ≈ 50,000  (random. Model knows nothing)
loss = 7.0:    perplexity = e^7.0  ≈ 1,100    (learning word frequencies)
loss = 5.0:    perplexity = e^5.0  ≈ 150      (learning basic grammar)
loss = 3.0:    perplexity = e^3.0  ≈ 20       (decent language model)
loss = 2.0:    perplexity = e^2.0  ≈ 7.4      (good language model)
loss = 1.5:    perplexity = e^1.5  ≈ 4.5      (very good)
loss = 1.0:    perplexity = e^1.0  ≈ 2.7      (excellent)
```

Perplexity gives you a mental model for what the loss actually means.
When your loss goes from 10.8 to 7.0 you have not just improved by
3.8 units. You have gone from being as confused as 50000 options to
being as confused as 1100 options. That is dramatic improvement.

## How to compute it

In code perplexity is one line.

```python
import math

loss = 3.0  # Your model's cross entropy loss
perplexity = math.exp(loss)

print(f"Loss: {loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
print(f"The model is as uncertain as picking among {perplexity:.0f} options.")
```

For a batch of predictions compute the average loss first then exponentiate.

```python
total_loss = 0
total_tokens = 0
for input_ids, target_ids in dataloader:
    with torch.no_grad():
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += target_ids.numel()

average_loss = total_loss / total_tokens
perplexity = math.exp(average_loss)
print(f"Validation perplexity: {perplexity:.2f}")
```

Always compute perplexity on data the model has not seen during
training. Training perplexity can be misleadingly low because the model
has memorized parts of the training data. Validation perplexity
measures how well the model generalizes.

## What different perplexity values mean

### Perplexity around 50000

Your model is random. It assigns equal probability to every token in
the vocabulary. It has learned nothing. This is normal at step zero
of training. If it stays here after thousands of steps something is
broken. Check your loss function and optimizer.

### Perplexity around 1000

The model has learned that some words are more common than others.
It knows that *the* appears often and *xylophone* appears rarely. It
uses these frequencies in its predictions. But it does not yet
understand word order or grammar or meaning. The output is gibberish
but the gibberish contains common words in roughly the right
proportions.

### Perplexity around 100

The model has learned basic grammar. It knows that articles precede
nouns. It knows that verbs agree with subjects in number. It knows
that periods end sentences. The output has recognizable sentence
structure even if the content is nonsensical. This is where most small
models plateau after limited training.

### Perplexity around 20

The model writes coherent text. Sentences have subjects and verbs and
objects in the right order. The content is sometimes factual and
sometimes invented. This is the level of GPT-1 from 2018. A model
with 17 million parameters trained on 100 million tokens might reach
this level.

### Perplexity around 10

The model writes good text. The content is mostly factual. Few obvious
errors. This is the level of GPT-2 from 2019. A model with 150 million
parameters trained on billions of tokens can reach this.

### Perplexity around 5

The model writes excellent text. Rarely makes factual errors. Handles
complex reasoning. This is the level of GPT-3 from 2020 and modern
small models like LLaMA 7B. Training these models costs millions of
dollars.

### Perplexity below 3

The model approaches human performance on language modeling. It
predicts what a human would write with high accuracy. Models at this
level are measured on harder tasks like question answering and code
generation because perplexity stops being a useful metric. The
difference between perplexity 2.5 and 2.3 is hard to feel but
expensive to achieve.

## Why perplexity is not everything

Perplexity measures how well the model predicts the next word. It does
not measure whether the model is helpful or truthful or safe or
creative. A model can have great perplexity and still generate
harmful content. It can have great perplexity and still hallucinate
facts. It can have great perplexity and still be boring.

Perplexity also depends on the dataset. A model trained on children's
books will have low perplexity on children's books and high perplexity
on legal documents. Perplexity is always relative to the test data.
Comparing perplexity between models is only meaningful when the models
are evaluated on the same dataset with the same tokenizer.

Different tokenizers produce different perplexity values for the same
model on the same data. A tokenizer with a larger vocabulary usually
gives lower perplexity because each token encodes more information and
there are fewer predictions to make per sentence. This is why you
cannot compare perplexity between models that use different tokenizers.

## The relationship to bits per character

Perplexity can be converted to bits per character. Bits per character
measures how many bits of information the model needs on average to
encode each character of text. Lower is better. The relationship is:

```
bits_per_character = ln(perplexity) / (ln(2) × characters_per_token)
```

For GPT-2 each token covers about 4 characters on average.

```
perplexity = 20:
  bits_per_char = ln(20) / (0.693 × 4) ≈ 1.08 bits per character

perplexity = 10:
  bits_per_char = ln(10) / (0.693 × 4) ≈ 0.83 bits per character
```

This tells you that the model needs about one bit of information per
character to encode English text. The theoretical minimum entropy of
English is about 0.6 to 1.0 bits per character. Models are approaching
that limit. Further improvements in perplexity will require better
understanding of meaning not just better statistics.

## Perplexity during training

A good training run should show perplexity decreasing smoothly. The
starting perplexity should be close to the vocabulary size. The final
perplexity depends on model size and data quality and training
duration.

```
Step       Loss    Perplexity
0        10.82     50,257     Model knows nothing
100       9.23     10,240     Learning frequencies
500       7.45      1,720     Learning word patterns
1,000     6.12        455     Emerging grammar
5,000     4.23         69     Coherent phrases
10,000    3.45         31     Decent sentences
50,000    2.89         18     Good model
```

If perplexity stops decreasing before step 10000 something is limiting
the model. The learning rate might be too low. The model capacity
might be exhausted. The data might not contain enough patterns to learn
from. Try increasing the learning rate or the model size or the dataset
size.

If perplexity decreases on training data but increases on validation
data the model is overfitting. It is memorizing the training set
instead of learning general patterns. The training and validation
curves diverge. Add more dropout or weight decay or use early stopping.

## Perplexity of famous models

All numbers are approximate and depend on the evaluation dataset.

```
Model                  Params     Perplexity (WikiText-103)
Random baseline        :          50,257
GPT-1 (2018)           117M       ~35
GPT-2 Small (2019)     124M       ~19
GPT-2 Medium            350M       ~15
GPT-2 Large             774M       ~12
GPT-3 Small (2020)      125M       ~18
GPT-3 XL               1.3B       ~10
GPT-3 6.7B                       ~8
GPT-3 175B                       ~5
LLaMA 7B (2023)          7B       ~7
LLaMA 13B                        ~6
LLaMA 70B                        ~4
```

Notice something. GPT-2 Small at 124 million parameters achieves
perplexity 19. GPT-3 at 125 million parameters achieves perplexity
18. Same architecture. Same size. Different training. GPT-3 was
trained on more data for longer. The extra compute improved perplexity
even without increasing model size. This is why data quality and
training duration matter as much as model architecture.

## What you need to remember

Perplexity is the exponentiated cross entropy loss. A perplexity of N
means the model is as uncertain as if it had to pick randomly among N
options. Lower is better. Random models start at around the vocabulary
size. Good models reach single digits.

Perplexity translates the abstract loss number into something visual.
When your loss drops from 10 to 5 your model has gone from being
confused among 22000 options to being confused among 150 options.
That is the difference between a model that knows nothing and a model
that knows something.

Perplexity does not measure helpfulness or truthfulness or safety.
It only measures how well the model predicts the next word. For
evaluating whether your model is useful you need other metrics. But
for tracking whether your model is learning perplexity is the single
most important number.
