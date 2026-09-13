# 14. Beyond the block

Three V4.1 features change the training procedure or the network's overall shape rather than the
block. None of them is present in owlet1.

Each of the three sits outside the scope of the previous chapters for a different reason. The first
adds a head above the stack and a term to the loss, leaving every block as it was. The second is a
separate small network trained after the main model is finished. The third rewires which layer's
output feeds which layer's key-value tables, which is a property of the stack rather than of any
block in it.

## Multi-token prediction

An ordinary language model is trained on one objective, the probability of the next token, and the
change here is to add a second one without touching the layers that produce it.

Add a small extra module at the top of the network that predicts token $t + 2$ from the state that
predicted $t + 1$, and train both with cross-entropy:

$$
\mathcal{L} = \mathcal{L}_{t+1} + \lambda\, \mathcal{L}_{t+2} .
$$

DeepSeek-V3 used this in pre-training as a denser training signal and then as a speculative-decoding
draft head at inference. V4.1 omits it during backbone pre-training.

Two separate benefits are claimed for the same module, and they are worth separating. As a training
signal, requiring the state at position $t$ to carry information about position $t + 2$ forces it to
represent more than the immediate continuation, which is the argument of Gloeckle et al. 2024,
"Better & Faster Large Language Models via Multi-token Prediction". As an inference device, the
second head is a draft head for speculative decoding.

The draft use rests on the cost structure of chapter 5. A decode step is limited by reading the
weights and the cache, not by arithmetic, so a forward pass that scores several candidate positions
at once costs almost the same as one that scores a single position. Speculative decoding, introduced
by Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding", and by Chen et
al. 2023, "Accelerating Large Language Model Decoding with Speculative Sampling", exploits this. A
cheap draft proposes the next few tokens, the expensive model scores all of them in one pass, and an
acceptance rule keeps the longest prefix that is consistent with what the expensive model would have
produced on its own. The accepted tokens are distributed exactly as the expensive model's own
sampling would have distributed them, so the output is unchanged and only the speed differs. The
multi-token head supplies the draft at almost no cost, since it reads a state the network has already
computed. What it buys depends entirely on how often its proposals are accepted.

## DSpark

The second feature is a separate network rather than a modification of the first, and it is trained
only after the main model is finished.

DSpark is a dedicated speculative decoder trained after pre-training with the backbone frozen. It
has three blocks with a 128-token window that draft five positions at a time. A Markov head and a
confidence head decide how many drafted tokens to submit for verification. The backbone verifies the
draft in one forward pass, so throughput rises without any change to the model's outputs.

Freezing the backbone is what makes this a post-training step rather than an architectural one. The
model being accelerated is fixed, its distribution is fixed, and the drafter is fitted to it. The
three blocks and the 128-token window keep the drafter cheap enough that five drafted positions cost
much less than the one backbone pass that verifies them. The two heads address the other half of the
trade. Submitting more drafted tokens raises the gain when they are accepted and wastes the tail of
the draft when they are not, so a confidence estimate of how far the draft can be trusted decides how
much of it to send. Because the verification rule preserves the backbone's distribution, none of this
changes what the model produces.

## Encoder-decoder split

The third feature changes which layers read from which, so it cannot be described by looking at any
single block.

V4.1 divides its layers into an encoder half and a decoder half. The decoder layers' global
key/value tables are computed from the encoder's last hidden state rather than from each decoder
layer's own input:

$$
C_\ell = H_{L/2}\, W^{KV}_\ell, \qquad \ell > L/2 .
$$

The decoder half then runs with global tables produced once, at the midpoint. This suits an
inference regime where the prompt is encoded once and many tokens are generated. The report calls
this the cross-encoder-decoder, or CED, design. owlet1 does not implement it. Every layer compresses
its own input.

The name refers back to the original transformer of Vaswani et al. 2017, "Attention Is All You Need",
which had a separate encoder stack whose output the decoder attended to through cross-attention, and
to the text-to-text models built on that shape such as Raffel et al. 2020, "Exploring the Limits of
Transfer Learning with a Unified Text-to-Text Transformer". Language models moved to decoder-only
stacks, in which every layer attends to its own input. The CED design brings the split back inside a
decoder-only stack by dividing it in half rather than by adding a second network. Sun et al. 2024,
"You Only Cache Once: Decoder-Decoder Architectures for Language Models", proposes the same
rearrangement, with a global key-value state produced once and reused by all later layers.

The serving argument is the one that matters. Prefill and decode have different cost profiles. The
prompt is processed once, in parallel over all its positions, and is compute-bound. Generation is
one position at a time and is bound by the bytes read per step, as chapter 5 established. Making the
decoder half's global tables a function of the midpoint state alone means those tables are built
during prefill and never rebuilt, so the decode path reads them rather than producing them, and the
compression work of chapter 9 disappears from the per-step cost for half the stack. It also means
that only one global table has to be stored per layer group in the decoder half, which is the
sharing that chapter 9 counted in the ledger.

The cost is that the decoder half cannot refine its global view. Its tables were computed from a
state produced at layer $L/2$, so whatever the second half of the network learns about the sequence
cannot be written back into what it attends over. The design bets that a representation formed by
40 blocks is a good enough summary for the remaining 40 to read.

The block is now complete and everything around it has been described. The next chapter puts the
pieces together into a single V4.1 block and totals the ledger over the whole book.
