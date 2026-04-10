# Pipeline Debug Log

This file records notebook pipeline errors in the order they appear during execution, along with the fix applied.

## 1. `argparse.Namespace` construction crash

### Where it appeared

- [TrainTools/train.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/TrainTools/train.py)
- Triggered when executing the training cell in [assignment1.ipynb](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/assignment1.ipynb)

### Error

`TypeError: Namespace.__init__() takes 1 positional argument but 2 were given`

### Cause

The code tried to construct `argparse.Namespace` by passing a dictionary as a single positional argument:

```python
argparse.Namespace({k: v for k, v in locals().items()})
```

`argparse.Namespace` expects keyword arguments, not a dictionary positional argument.

### Fix

Kept the original line commented as old code and replaced it with keyword expansion:

```python
# old code
# args = argparse.Namespace({k: v for k, v in locals().items()})
args = argparse.Namespace(**{k: v for k, v in locals().items()})
```

### Why this fix works

Using `**` expands the dictionary into named fields, which is exactly how `Namespace` is meant to be built.

## 2. Positional encoding shape mismatch

### Where it appeared

- [Models/encoder.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/encoder.py)
- Triggered during `QANet(...)` construction from [TrainTools/train.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/TrainTools/train.py)

### Error

`RuntimeError: The size of tensor a (400) must match the size of tensor b (96) at non-singleton dimension 1`

### Cause

`PosEncoder` builds tensors for the sinusoidal encoding with incompatible shapes before multiplying them. The length dimension and model dimension are arranged inconsistently, so broadcasting fails during:

```python
pe = torch.sin(pos * freqs + phases)
```

### Fix

Kept the original tensor construction commented as old code and rebuilt the tensors with broadcast-compatible shapes:

```python
# old code
# freqs = ...
# phases = ...
# pos = ...
freqs = ... .unsqueeze(1)   # [C, 1]
phases = ... .unsqueeze(1)  # [C, 1]
pos = torch.arange(length, dtype=torch.float32).unsqueeze(0)  # [1, L]
```

### Why this fix works

The positional encoding formula needs channel-wise terms shaped like `[C, 1]` and positions shaped like `[1, L]` so broadcasting produces `[C, L]`. The previous version arranged one tensor as `[C, L]` and another as `[1, C]`, which cannot multiply elementwise.

## 3. Unknown scheduler name from notebook config

### Where it appeared

- [assignment1.ipynb](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/assignment1.ipynb)
- Validated inside [TrainTools/train.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/TrainTools/train.py)

### Error

`ValueError: Unknown scheduler 'none'. Available: ['cosine', 'step', 'lambda']`

### Cause

The notebook training cell requests `scheduler_name="none"`, but the scheduler registry did not define a `"none"` entry.

### Fix

Added an explicit `none_scheduler()` alias in [Schedulers/scheduler.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Schedulers/scheduler.py) and registered it as `"none"`. The implementation uses the same constant-learning-rate behavior as the existing lambda scheduler.

### Why this fix works

It preserves the notebook’s intended configuration without forcing notebook edits, and it makes `"none"` a valid scheduler name that simply leaves the learning rate unchanged across steps.

## 4. Word and character embeddings were swapped in `QANet.forward()`

### Where it appeared

- [Models/qanet.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/qanet.py)
- Triggered on the first training batch from [TrainTools/train_utils.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/TrainTools/train_utils.py)

### Error

`IndexError: index out of range in self`

### Cause

The forward pass sent context word ids into `self.char_emb` and context character ids into `self.word_emb`. Character vocabularies are much smaller than word vocabularies, so word ids immediately exceed the valid range of the character embedding table.

### Fix

Kept the original lines commented as old code and swapped the lookups back to the correct tables:

```python
# old code
# Cw, Cc = self.char_emb(Cwid), self.word_emb(Ccid)
# Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
```

### Why this fix works

Word ids must be looked up in the word embedding matrix, and character ids must be looked up in the character embedding matrix. That restores the expected tensor shapes and valid index ranges for the embedding stage.

## 5. Character embedding tensor was permuted in the wrong order

### Where it appeared

- [Models/embedding.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/embedding.py)
- Triggered inside the character convolution path from [Models/conv.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/conv.py)

### Error

`RuntimeError: shape '[1, 64, 16, 1, 1]' is invalid for input of size 4096`

### Cause

The character embedding input starts as `[B, L, char_len, d_char]`, but it was permuted to `[B, char_len, L, d_char]` instead of `[B, d_char, L, char_len]`. That made the convolution think the number of input channels was `char_len` rather than `d_char`, which broke the grouped convolution weight reshape.

### Fix

Kept the original line commented as old code and changed the permutation to:

```python
# old code
# ch_emb = ch_emb.permute(0, 2, 1, 3)
ch_emb = ch_emb.permute(0, 3, 1, 2)
```

### Why this fix works

The 2D convolution expects the channel dimension second. Moving `d_char` into axis 1 restores the expected `[B, d_char, L, char_len]` layout for character convolution and max pooling.

## 6. `Conv2d` width padding used the stale height value

### Where it appeared

- [Models/conv.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/conv.py)
- Triggered in the character convolution path used by [Models/embedding.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/embedding.py)

### Error

`RuntimeError: Sizes of tensors must match except in dimension 3. Expected size 400 but got size 404 for tensor number 1 in the list.`

### Cause

Inside `Conv2d.forward()`, the code first pads the height dimension, which changes the tensor height from `H` to `H + 2p`. It then allocates width-padding using the old `H`, so `pad_w` and `x` no longer agree along dimension 2.

### Fix

Kept the original line commented as old code and replaced `H` with the current padded height:

```python
# old code
# pad_w = x.new_zeros(B, C_in, H, p)
pad_w = x.new_zeros(B, C_in, x.size(2), p)
```

### Why this fix works

After height padding, the current tensor height is `x.size(2)`. Using that value makes the width-padding tensor match the already-padded tensor, so concatenation along the width dimension is valid.

## 7. `Highway` transposed the batch dimension instead of the feature dimension

### Where it appeared

- [Models/embedding.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/embedding.py)
- Triggered inside the embedding fusion stage before the encoder blocks

### Error

`RuntimeError: mat1 and mat2 shapes cannot be multiplied (145600x8 and 364x364)`

### Cause

The highway layer receives `x` as `[B, C, L]` and should convert it to `[B, L, C]` before applying `nn.Linear(size, size)` over the feature dimension `C`. Instead, it used `transpose(0, 2)`, which produced `[L, C, B]`, leaving the last dimension equal to batch size.

### Fix

Kept the original line commented as old code and changed the transpose to swap channel and length axes:

```python
# old code
# x = x.transpose(0, 2)
x = x.transpose(1, 2)
```

### Why this fix works

`nn.Linear` operates on the last dimension. After `transpose(1, 2)`, the tensor shape is `[B, L, C]`, so the last dimension correctly matches the highway layer input size.

## 8. `Conv1d` unfolded over channels instead of sequence length

### Where it appeared

- [Models/conv.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/conv.py)
- Triggered by the context projection convolution in [Models/qanet.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/qanet.py)

### Error

`RuntimeError: shape '[8, 364, 0, 400, 5]' is invalid for input of size 1486720`

### Cause

`Conv1d.forward()` should extract sliding windows along the sequence dimension `L`, which is dimension `2` for an input shaped `[B, C_in, L]`. The code unfolded dimension `1`, which is the channel axis, so the resulting tensor layout was nonsensical for grouped convolution.

### Fix

Kept the original line commented as old code and changed the unfold dimension:

```python
# old code
# x_unf = x.unfold(1, self.kernel_size, 1)
x_unf = x.unfold(2, self.kernel_size, 1)
```

### Why this fix works

Convolution windows must slide across the sequence length, not across channels. Unfolding dimension `2` produces the expected `[B, C_in, L_out, k]` tensor for the subsequent grouped multiply-accumulate.

## 9. `DepthwiseSeparableConv` applied pointwise before depthwise

### Where it appeared

- [Models/conv.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/conv.py)
- Triggered during the context projection convolution in [Models/qanet.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/qanet.py)

### Error

`RuntimeError: shape '[8, 364, 0, 400, 5]' is invalid for input of size 1536000`

### Cause

In a depthwise-separable convolution, the depthwise operation must run first while the input still has its original channel count. The implementation ran the pointwise convolution first, reducing the channel dimension from `364` to `96`, and then fed that tensor into a depthwise layer configured with `groups=364`.

### Fix

Kept the original line commented as old code and reversed the order:

```python
# old code
# return self.depthwise_conv(self.pointwise_conv(x))
return self.pointwise_conv(self.depthwise_conv(x))
```

### Why this fix works

The depthwise layer now processes the original input channels it was configured for, and the pointwise layer then mixes those channels into the requested output dimension.

## 10. `LayerNorm` reduced away broadcast dimensions and applied affine terms in the wrong order

### Where it appeared

- [Models/Normalizations/layernorm.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/Normalizations/layernorm.py)
- Triggered when the first encoder block normalized the context representation

### Error

`RuntimeError: The size of tensor a (400) must match the size of tensor b (8) at non-singleton dimension 2`

### Cause

The mean and variance were computed with `keepdim=False`, so their shapes collapsed from `[B, C, L]` to `[B]` when normalizing over the last two dimensions. That shape cannot broadcast back over the original tensor. Also, the affine transform used `x_norm * bias + weight` instead of the standard `x_norm * weight + bias`.

### Fix

Kept the original lines commented as old code and changed them to:

```python
# old code
# mean = x.mean(dim=dims, keepdim=False)
# var = x.var(dim=dims, keepdim=False, unbiased=False)
mean = x.mean(dim=dims, keepdim=True)
var = x.var(dim=dims, keepdim=True, unbiased=False)

# old code
# return x_norm * self.bias + self.weight
return x_norm * self.weight + self.bias
```

### Why this fix works

`keepdim=True` preserves singleton dimensions so the normalized statistics broadcast correctly over `[B, C, L]`. Using `weight` as the multiplicative scale and `bias` as the additive shift restores the standard layer normalization formula.

## 11. Encoder block normalization indexed one element past the end

### Where it appeared

- [Models/encoder.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/encoder.py)
- Triggered inside the first encoder block during the convolution stack

### Error

`IndexError: index 4 is out of range`

### Cause

`self.norms` is created with exactly `conv_num` normalization layers, indexed `0` through `conv_num - 1`. The forward loop used `self.norms[i + 1]`, so the final iteration tries to access index `conv_num`, which does not exist.

### Fix

Kept the original line commented as old code and changed the lookup to:

```python
# old code
# out = self.norms[i + 1](out)
out = self.norms[i](out)
```

### Why this fix works

Each convolution iteration now uses the corresponding normalization layer that was actually created for that position in the stack.

## 12. Context and question masks were passed to `CQAttention` in reverse order

### Where it appeared

- [Models/qanet.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/qanet.py)
- Triggered when entering the context-query attention stage

### Error

`RuntimeError: The size of tensor a (400) must match the size of tensor b (50) at non-singleton dimension 2`

### Cause

`CQAttention.forward()` expects arguments in the order `(C, Q, cmask, qmask)`, but `QANet.forward()` passed `(Ce, Qe, qmask, cmask)`. That swapped the context and question masks, causing shape mismatch when masking the similarity tensor `S` of shape `[B, Lc, Lq]`.

### Fix

Kept the original line commented as old code and passed the masks in the expected order:

```python
# old code
# X = self.cq_att(Ce, Qe, qmask, cmask)
X = self.cq_att(Ce, Qe, cmask, qmask)
```

### Why this fix works

The context mask now aligns with the context length dimension and the question mask aligns with the question length dimension, so masking and softmax operate on compatible shapes.

## 13. `CQAttention` multiplied the question tensor and attention weights in the wrong order

### Where it appeared

- [Models/attention.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/attention.py)
- Triggered immediately after the mask-order fix when computing attended question vectors

### Error

`RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [8, 96] but got: [8, 400].`

### Cause

`Q` has shape `[B, Lq, C]` and `S1` has shape `[B, Lc, Lq]`. To compute the attended question representation for each context position, the multiplication must be `S1 @ Q`, producing `[B, Lc, C]`. The code used `Q @ S1`, which is dimensionally incompatible.

### Fix

Kept the original line commented as old code and reversed the multiplication order:

```python
# old code
# A = torch.bmm(Q, S1)
A = torch.bmm(S1, Q)
```

### Why this fix works

Each context position now uses its attention weights over the question tokens to form a weighted sum of question representations, yielding the expected `[B, Lc, C]` tensor.

## 14. Pointer head concatenated `M1` and `M2` along the batch axis

### Where it appeared

- [Models/heads.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/heads.py)
- Triggered when computing answer start and end logits

### Error

`RuntimeError: size mismatch, got input (6400), mat (6400x96), vec (192)`

### Cause

The pointer head expects to concatenate model states along the channel dimension to produce `[B, 2C, L]`. `X1` was concatenated with `dim=0`, which doubles the batch dimension instead and leaves the channel dimension unchanged, making `self.w1` incompatible with the resulting tensor.

### Fix

Kept the original line commented as old code and changed the concatenation axis:

```python
# old code
# X1 = torch.cat([M1, M2], dim=0)
X1 = torch.cat([M1, M2], dim=1)
```

### Why this fix works

The pointer parameters `w1` and `w2` are sized for `2 * d_model` channels. Concatenating along `dim=1` restores the intended `[B, 2C, L]` representation before projecting to start/end logits.

## 15. `qa_nll_loss` passed inputs and targets to `nll_loss` in the wrong order

### Where it appeared

- [Losses/loss.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Losses/loss.py)
- Triggered on the first training batch after the model produced start/end log-probabilities

### Error

`RuntimeError: 0D or 1D target tensor expected, multi-target not supported`

### Cause

`torch.nn.functional.nll_loss` expects arguments in the order `(input, target)`. The start-loss term was written as `F.nll_loss(y1, p1)`, reversing the prediction tensor and target tensor.

### Fix

Kept the original line commented as old code and corrected the argument order:

```python
# old code
# return 0.5 * (F.nll_loss(y1, p1) + F.nll_loss(p2, y2))
return 0.5 * (F.nll_loss(p1, y1) + F.nll_loss(p2, y2))
```

### Why this fix works

`p1` and `p2` are the model’s log-probability tensors over positions, while `y1` and `y2` are the 1D gold start/end indices. Passing `(prediction, target)` restores the expected NLL loss interface.

## 16. Backpropagation was called on `loss.item()` instead of the tensor loss

### Where it appeared

- [TrainTools/train_utils.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/TrainTools/train_utils.py)
- Triggered on the first training iteration after loss computation

### Error

`AttributeError: 'float' object has no attribute 'backward'`

### Cause

`loss.item()` converts the PyTorch scalar tensor into a plain Python float for logging. Once converted, it no longer carries the computation graph, so calling `.backward()` on it is invalid.

### Fix

Kept the original line commented as old code and backpropagated through the tensor itself:

```python
# old code
# loss.item().backward()
loss.backward()
```

### Why this fix works

`loss` is the actual scalar tensor produced by the model and loss function, so it still tracks gradients and can propagate them through the network.

## 17. Checkpoint saving failed because the scheduler stored an unpicklable local lambda

### Where it appeared

- [Schedulers/scheduler.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Schedulers/scheduler.py)
- Surfaced when [TrainTools/train_utils.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/TrainTools/train_utils.py) tried to save the checkpoint after the first train/dev evaluation cycle

### Error

`_pickle.PicklingError: Can't pickle local object <function none_scheduler.<locals>.<lambda> ...>`

### Cause

The `none` scheduler was implemented with an inline lambda passed into `LambdaLR`. When PyTorch serialized the scheduler state, it tried to pickle that lambda, but locally defined lambda functions are not picklable.

### Fix

Added a top-level helper function and reused it for both constant-learning-rate schedulers:

```python
def _constant_lr_factor(_):
    return 1.0

# old code
# return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
return LambdaLR(optimizer, lr_lambda=_constant_lr_factor)
```

### Why this fix works

Top-level functions are picklable, so the scheduler state can now be serialized safely inside checkpoints.

## 18. `torch.load` failed under PyTorch's default `weights_only=True` behavior

### Where it appeared

- [EvaluateTools/evaluate.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/EvaluateTools/evaluate.py)
- Triggered when loading the saved checkpoint during evaluation

### Error

`_pickle.UnpicklingError: Weights only load failed ...`

### Cause

In newer PyTorch versions, `torch.load()` defaults to `weights_only=True`. The saved checkpoint contains more than raw tensor weights, including scheduler metadata, so the restricted loader rejects it.

### Fix

Kept the original line commented as old code and explicitly disabled the restricted loading mode:

```python
# old code
# ckpt = torch.load(ckpt_path, map_location=DEVICE)
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
```

### Why this fix works

It restores the older full-checkpoint loading behavior that this code expects, allowing the serialized dictionary to be read back completely.

## 19. Evaluation expected the wrong checkpoint key for model weights

### Where it appeared

- [EvaluateTools/evaluate.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/EvaluateTools/evaluate.py)
- Triggered right after the checkpoint was successfully loaded

### Error

`KeyError: 'model'`

### Cause

The checkpoint writer in [TrainTools/train_utils.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/TrainTools/train_utils.py) saves weights under the key `model_state`, but evaluation tried to read `ckpt["model"]`.

### Fix

Kept the original line commented as old code and added a compatibility fallback:

```python
# old code
# model.load_state_dict(ckpt["model"])
model_state = ckpt["model"] if "model" in ckpt else ckpt["model_state"]
model.load_state_dict(model_state)
```

### Why this fix works

Evaluation can now load both checkpoint formats, which makes it compatible with the training code currently used in this repository.

## 20. Custom dropout used the wrong scaling factor and caused `nan` activations

### Where it appeared

- [Models/dropout.py](/Users/benjaminvasquez/Documents/MSc%20Data%20Science/DL%20COMP4329%20COMP5329/Assignments/Assignment%201/Assignment1_old/Models/dropout.py)
- Triggered during the first training-mode forward pass before any optimizer update

### Error

No hard exception occurred here, but the first training forward pass already produced `nan` values in `p1`, `p2`, and the loss.

### Cause

The custom inverted dropout implementation scaled surviving activations by `1 / p` instead of `1 / (1 - p)`. With dropout rates like `0.05` and `0.1`, this amplified activations by factors of `20` and `10`, which quickly destabilized the forward pass and produced `nan` outputs.

### Fix

Kept the original line commented as old code and corrected the scaling:

```python
# old code
# return x * mask / self.p
return x * mask / (1.0 - self.p)
```

### Why this fix works

In inverted dropout, activations are scaled by the keep probability, not the drop probability. Using `1 / (1 - p)` preserves the expected activation magnitude during training and avoids the severe over-amplification that caused numerical instability.
