# How to Read a Loss Curve

## The short answer

A loss curve is the single most important diagnostic for training a
language model. It tells you if the model is learning and if it is
learning well and if something is broken. A good curve goes down
smoothly. That is the happy path. Everything else is a problem.
This guide shows you every common pattern and what to do about it.

## The setup

You are training a model. Every 100 steps you record the loss. You
plot the steps on the x axis and the loss on the y axis. You want
the line to go down. But the shape of the line matters more than the
direction. Two training runs can both have decreasing loss but one
is doomed and the other is thriving.

All the examples below use starting loss around 10.8 which is the
loss of a random model predicting uniformly over GPT-2's 50257
token vocabulary. Loss equals negative log of one divided by 50257
which is approximately ln(50257) which is approximately 10.82.

## Pattern 1: The good curve

```
Loss
 10.8 |*
      | *
  9.0 |  *
      |   *
  7.0 |    *
      |     **
  5.0 |       ***
      |          ******
  3.0 |                ************
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
                    Steps
```

The loss starts around 10.8. It drops quickly in the first few
thousand steps. The drop slows as training continues. The curve is
smooth. No spikes. No plateaus. This is what successful training
looks like.

The early rapid drop is the model learning basic patterns. Capital
letters start sentences. Periods end them. Common words appear in
predictable positions. These patterns are easy to learn because
they are consistent. The loss drops fast.

The later slow descent is the model learning subtle patterns.
Subject verb agreement. Pronoun resolution. The difference between
affect and effect. These patterns are harder because they depend
on long range context and nuanced meaning.

The curve never plateaus completely because the model always has
something more to learn. There is always a slightly better set of
weights that predicts the next word a tiny bit more accurately.

## Pattern 2: The flat line

```
Loss
 10.8 |********************************************
      |
  8.0 |
      |
  5.0 |
      |
  2.0 |
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
```

The loss never moves. It stays at 10.8 forever. The model is not
learning at all. This is almost always a bug.

Possible causes in order of likelihood.

The learning rate is zero or the optimizer is not stepping. Check
that `optimizer.step()` is called. Check that `optimizer.zero_grad()`
is called AFTER step not before.

The gradients are zero. A bug in the loss computation. Check that
the logits and targets are correctly aligned. The logits should be
for predicting the next token. The targets should be the actual
next tokens.

The model weights are not updating. Check `weight.grad` after
`loss.backward()`. It should be non-zero for at least some weights.

The data is broken. Maybe input and target are identical. Maybe all
targets are the same token. Print a few batches and inspect them.

## Pattern 3: Loss spikes

```
Loss
 10.8 |*
      | *
  9.0 |  *
      |   *      /
  7.0 |    *    /
      |     ** /
  5.0 |    /  **
      |   /      ***
  3.0 |  /           ******
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
                    ↑
                   spike
```

The loss was decreasing nicely. Then it jumped up by a factor of
two or three. The model is still learning but something bad happened.

Cause: gradient explosion. A rare batch of training data contained
an unusual pattern. The gradients became very large. The weights took
a large step in a bad direction.

Fix: add gradient clipping. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Call this after `loss.backward()` and before `optimizer.step()`. Our training code already does this.

If clipping is already in place the max norm might be too high. Try
reducing from 1.0 to 0.5. If the spikes persist lower the learning
rate.

A single spike is not fatal. The model usually recovers. But repeated
spikes are a sign that something is systematically wrong. The data
might contain corrupted examples. The learning rate might be too
high. The architecture might have a bug in how it handles certain
sequence lengths.

## Pattern 4: The plateau

```
Loss
 10.8 |*
      | *
  9.0 |  *
      |   *
  7.0 |    *
      |     *
  6.5 |      ******************************
      |
  3.0 |
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
               ↑
          plateau starts
```

The loss drops to around 6.5 and then stops. No matter how many more
steps you train the loss refuses to go lower. The model has hit a
wall.

This is called a learning rate plateau. The learning rate is too low
for the model to escape its current region of weight space. The
gradients are too small to move the model to a better configuration.

Fix: increase the learning rate. If you are using the cosine schedule
try increasing the peak learning rate from 3e-4 to 1e-3. If you are
already at the minimum of the schedule restart with a higher peak.

Alternative cause: the model capacity is exhausted. A model with 17
million parameters trained on 100 megabytes of text can only capture
so much. The loss cannot go below a certain floor determined by the
model size and data complexity. To push lower you need a bigger model
or more data or both.

## Pattern 5: Training loss goes down but validation loss goes up

```
Loss
 10.8 |*
  9.0 | *    Training (solid)
  7.0 |  *         *
      |   *       * *
  5.0 |    *     *   *
      |     *   *     *
  3.0 |      ***       ************* Validation (dashed)
      |        *                     **************
  1.0 |         *****                               ******
      +-----+-----+-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k   35k   40k
                   ↑
              overfitting starts
```

The training loss keeps decreasing. The validation loss was decreasing
too but then it started going up. The model is overfitting. It is
memorizing the training data instead of learning general patterns.

Fix: stop training at the point where validation loss starts rising.
This is called early stopping. You do not need to fix the model. You
just need to stop before it overfits.

Prevention: increase dropout from 0.1 to 0.2 or 0.3. Increase weight
decay from 0.1 to 0.2. Use a larger and more diverse training dataset.
The model cannot memorize data it has not seen enough times.

## Pattern 6: Loss goes negative or NaN

```
Loss
 10.8 |*
      | *
  9.0 |  *
      |   *
  7.0 |    *    |
      |     *   |
  5.0 |      *  |
      |       * |
  3.0 |        *|
      |         *
  0.0 +----------*--------*-------*------
  --------------------------------
  NaN  NaN  NaN  NaN  NaN  NaN  NaN
```

The loss was decreasing. Then it went to zero. Then it became NaN.
Training is dead. The model is producing infinities or NaN values.

Cause: numerical overflow. The logits or the loss computation produced
numbers too large for bfloat16 or float32 to represent. This happens
when the learning rate is too high and the weights blow up.

Fix: lower the learning rate dramatically. Add gradient clipping.
Check that you are dividing by `sqrt(head_dim)` in the attention
scores. Without this division the scores can become large enough that
`exp(score)` overflows bfloat16. Check that your loss computation is
correct. Cross entropy with logits of 1000 and targets produces NaN.

If using mixed precision check that the autocast context is set up
correctly. Some operations like softmax and layer norm should stay
in float32 for numerical stability. Autocast handles this but only
if the backend supports it.

## Pattern 7: The step function

```
Loss
 10.8 |*
      | *
  9.0 |******
      |      ******
  7.0 |           ******
      |                ******
  5.0 |                     ******
      |                          ******
  3.0 |                               ******
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
```

The loss drops suddenly at regular intervals. The drops coincide with
learning rate changes. The learning rate was reduced by a factor of
10 and the loss jumped down.

Cause: step decay learning rate schedule. Some training setups use
this intentionally. But the sudden drops can disturb the model's
momentum. The model took a while to adapt to the old learning rate
and now the rate changed abruptly.

Preference: use cosine decay instead of step decay. Cosine decay is
smooth. The model adapts continuously. No sudden jumps. Our training
code uses cosine warmup with cosine decay. No step changes.

## Pattern 8: The sawtooth

```
Loss
 10.8 |*  *  *  *
      | *  *  *  *
  9.0 |*  *  *  *
      | *  *  *  *
  7.0 |*  *  *  *  *  *
      | *  *  *  *  *  *
  5.0 |*  *  *  *  *  *  *  *
      | *  *  *  *  *  *  *  *
  3.0 |*  *  *  *  *  *  *  *  *  *
      +-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k
```

The loss bounces up and down every few steps. The overall trend is
downward but the noise is high.

Cause: batch size is too small. Each batch is too small to be
representative of the overall data distribution. Some batches are
easy and produce low loss. Others are hard and produce high loss.
The model overreacts to each batch.

Fix: increase the batch size or use gradient accumulation to simulate
a larger batch. Our training code uses gradient accumulation with
`grad_accum_steps=2`. Try increasing to 4 or 8. The effective batch
size increases without using more GPU memory.

The sawtooth is not necessarily a problem. Even with a good batch
size some noise is normal. The model learns the average gradient
direction over many batches. Individual batches can be noisy as long
as the average is correct.

## What you need to remember

A good loss curve is smooth and decreasing with a steep early drop
and a gradual later descent. No spikes. No plateaus. No NaN.

If your curve does not look like Pattern 1 something is wrong. The
most common fixes are gradient clipping and learning rate adjustment
and batch size increases and dropout increases. Try them in that
order.

Always save your loss values during training. A loss curve is a
debugging tool. Without it you are training blind. With it you can
diagnose most training problems in seconds by matching the pattern
to one of the eight patterns above.
