# 6. Mixture of experts

Chapters 2 to 5 changed the dense block one part at a time. Chapter 5 was the first of them to
attack a cost, and the cost it attacked was memory at inference. This chapter attacks the other
cost, the weights. In a dense model every parameter is read and used for every token, so parameter
count and compute per token are the same quantity measured twice. Quality scales with parameters, so
a better model is a slower model, in training and in serving alike, in direct proportion. Every
design in this chapter exists to break that proportion.

## Where the parameters are

Chapter 1 counted the block. Per block the four attention matrices hold $4D^2 = 268$ million
parameters and the two feed-forward matrices hold $8D^2 = 537$ million, so one block holds 805
million. Over $L = 80$ blocks that is 64.4 billion, and the embedding adds $V D = 262$ million, for
about 65 billion in total. Two thirds of the model is feed-forward, and that fraction is the same for
every model of the GPT-2 shape, because it follows from the ratio of $8D^2$ to $4D^2$ and not from
the width.

Compute divides the same way. At about two floating-point operations per parameter, one token costs
about 130 GFLOPs from the weights, of which about 86 GFLOPs is feed-forward. Per block the dense
feed-forward layer costs 1.07 GFLOPs per token. Anything that raises the parameter count of the model
without raising this number has to act on this layer, because this is where the parameters are.

## Why the feed-forward layer is where the two can be separated

Attention mixes positions. Its parameters are tied to the head structure, its output for one token
depends on every earlier token, and its cost per token grows with the length of the sequence rather
than with its own size. The feed-forward layer is different. It acts on one token at a time, with no
interaction between positions, and its output for a token is a function of that token's state alone.
A different subset of its parameters can be applied to each token without changing the shape of
anything else in the block and without changing what the attention layers on either side see. The
layer is a pure function of one vector, so it can be replaced by a choice among many pure functions
of one vector.

Chapter 4 also gave a reading of what those parameters hold. Geva et al. 2021, "Transformer
Feed-Forward Layers Are Key-Value Memories", describe the layer as a store of key-value memories,
each row of the first projection a key matched against the input by a dot product and the
corresponding column of the second projection the value written back to the residual stream. A store
of that kind is used sparsely by construction. Most keys match a given token weakly and contribute
little to the output. Computing all of them for every token spends arithmetic in proportion to the
size of the store rather than in proportion to how much of it one token needs. A model that could
retrieve only the relevant part of the store would pay for what it uses.

## Conditional computation before the transformer

The idea is older than the transformer. Jacobs et al. 1991, "Adaptive Mixtures of Local Experts",
trained several small networks together with a gating network that weighted their outputs, so that
each expert specialised on a region of the input space. That mixture was dense, because every expert
ran on every input, and the saving was in learning rather than in compute.

Shazeer et al. 2017, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
Layer", made the mixture sparse and put it between the layers of an LSTM language model. A gating
network scored thousands of small feed-forward experts and kept only the few highest, so a model with
up to 137 billion parameters ran at the cost of a small one. That paper also introduced the two
problems the rest of this chapter is about: the gate collapses onto a few experts unless it is
corrected, and the tokens of a batch must be regrouped by expert before any matmul can run.

Lepikhin et al. 2020, "GShard: Scaling Giant Models with Conditional Computation and Automatic
Sharding", moved the layer into the transformer, replacing the feed-forward layer of every second
block with a top-2 mixture and sharding the experts across devices to reach 600 billion parameters
on a translation model. Fedus et al. 2021, "Switch Transformers: Scaling to Trillion Parameter Models
with Simple and Efficient Sparsity", simplified the gate to a single expert per token, argued that
top-1 is enough when the number of experts is large, and trained a 1.6 trillion parameter model.
Jiang et al. 2024, "Mixtral of Experts", brought the design to open-weight decoder models with 8
experts per layer and 2 active, about 47 billion parameters in total and about 13 billion used per
token. Dai et al. 2024, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts
Language Models", changed the shape of the expert set in the two ways that the current generation
uses, covered in [Fine-grained and shared experts](experts.md).

## Routed experts

Replace the single FFN with $E$ copies, called experts, and a small router that picks $K \ll E$ of
them per token:

$$
\operatorname{MoE}(x) = \sum_{i \in \mathcal{I}(x)} w_i(x)\, \operatorname{FFN}_i(x),
\qquad |\mathcal{I}(x)| = K .
$$

$\mathcal{I}(x)$ is the set of experts chosen for this token and $w_i(x)$ is the weight given to
each. The sum runs over $K$ terms, not $E$. Parameters grow by a factor of $E$, compute per token
grows only by a factor of $K$, and the ratio $E/K$ is the factor by which the two have been
separated. A model can hold the parameters of a very large dense model and pay the per-token cost of
one $E/K$ times smaller. The experts are ordinary SwiGLU blocks of the form given in chapter 4, so
nothing inside an expert is new. What is new is the router, the training corrections it needs, and
the data movement the choice implies.

Clark et al. 2022, "Unified Scaling Laws for Routed Language Models", fitted scaling laws to routed
models and found that increasing the number of experts improves the loss in a way that behaves like
an increase in parameters at fixed compute, with diminishing returns as the expert count grows. The
practical reading is that the substitution is favourable but not free, and that the routed parameter
is worth less than the dense parameter it stands in for.

## What it buys on the reference model

Take the layer shape of DeepSeek-V3 at the reference width: 256 routed experts of hidden width 2048
plus one shared expert, with $K = 8$. Each expert is a SwiGLU block of three matrices, so it holds
$3 \cdot 8192 \cdot 2048 = 50.3$ million parameters. All 257 of them hold

$$
257 \cdot 3 \cdot 8192 \cdot 2048 = 12.9 \text{ billion parameters per block},
$$

against 537 million for the dense layer. The nine experts a token passes through, the eight routed
and the one shared, hold $9 \cdot 3 \cdot 8192 \cdot 2048 = 453$ million, so compute per token is
about what the dense layer cost and slightly less. Over 80 blocks the feed-forward parameters come to
about 1 trillion, of which 36 billion are active per token. The layer holds 24 times the parameters
of the layer it replaces and does the same arithmetic. That ratio, and not any change inside the
expert, is the whole content of the design.

| layer | params per block | active per token per block | params over 80 blocks | active |
| --- | --- | --- | --- | --- |
| dense SwiGLU FFN | 537 M | 537 M | 43 B | 43 B |
| 256 routed of width 2048, plus 1 shared, $K = 8$ | 12.9 B | 453 M | 1.03 T | 36 B |

## Grouping the tokens by expert

The first cost is that the layer is no longer a matmul. A batch arrives as $N$ token states of width
$D$. In a dense layer those $N$ rows go through one matrix multiplication. In a routed layer each row
has its own list of $K$ destinations, so the batch must be expanded to $N K$ routed copies, sorted by
expert, multiplied expert by expert, and scattered back into token order with the weights $w_i$
applied. The sort is a gather with an index vector, and the multiplication is a grouped matmul over
$E$ matrices with a different number of rows on each. Implementations either loop over experts or use
a kernel that takes the group boundaries as an argument.

The sizes of the groups are decided by the router and change at every step. Two consequences follow.
Group sizes that are small or uneven make the matmuls inefficient, because a matmul with few rows on
one side is memory bound in the same way as decode attention in chapter 5. And a group that is far
larger than average sets the time of the whole layer. This is why load balancing, treated in
[Load balancing](balancing.md), is a throughput problem before it is a quality problem.

The reference implementation in this repository takes the simple route. `src/mzoo/archs/owlet1/moe.py`
counts the tokens per expert with `bincount`, skips empty experts, and gathers each expert's tokens
with a `torch.where`, one Python iteration per expert. That is readable and correct and is not a
serving path.

## Expert parallelism and the all-to-all

The second cost is that the expert weights must all be resident even though each token uses few of
them. One trillion feed-forward parameters at two bytes each is 2 TB, which no single accelerator
holds, so the experts of a layer are partitioned across devices. Each device holds a slice of the
experts of every layer and a full copy of everything that is not an expert. This is expert
parallelism, and it was introduced with GShard.

Under expert parallelism the grouping above becomes a network operation. A token whose chosen expert
lives on another device must have its state of width $D$ sent to that device and the result sent
back, so every mixture-of-experts layer contains two collective communications, a dispatch before the
experts and a combine after them. Both are all-to-all exchanges, because in general every device has
tokens for every other device. On the reference layer, in the worst case where none of a token's
eight experts is local, dispatch moves $8 \cdot 8192 \cdot 2 = 131$ KB per token and combine moves
the same again, and over 80 layers that is about 10 MB per token in each direction. Real systems move
much less than this, because some experts are local and because routing can be restricted to a
bounded number of devices, but the order of the number explains why the communication schedule is
part of the model design and why the router is not free to send a token anywhere.

The rest of the chapter takes the three parts of the design in turn.

Sections:

- [Routing](routing.md): how $\mathcal{I}(x)$ and $w_i$ are computed, and the DeepSeek score function.
- [Fine-grained and shared experts](experts.md): the DeepSeekMoE layout.
- [Load balancing](balancing.md): why routers collapse and the two fixes.
