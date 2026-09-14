# Writing notes

Observations drawn from chapter 3, offered as ideas rather than rules. Nothing here is
binding — each chapter should make its own call based on what it needs to say.

## Openings and endings

1. The chapter 3 opening does three things in two paragraphs: recap what earlier chapters
   settled, name why that arrangement runs out, and state plainly "This chapter replaces
   it." Consider whether your chapter earns the same directness.
2. A `Roadmap` section right after the opening can orient the reader: one item per
   section, each a bolded title plus one sentence on why the section matters. Number the
   items rather than using bullets, so any item can be referenced by number in a prompt
   ("roadmap item 3"). Optional but cheap — it doubles as a checklist that the chapter
   delivered what it promised.
3. Endings in this book tend to declare a topic settled (or name what remains open) and
   then pivot to the next chapter with a concrete hook — a number or a stake, not
   "in conclusion".

## Argumentation

1. Lead with the failure mode, not the fix. "Why a table of positions fails" comes before
   rotary embedding, and the fix reads better because the defect is already concrete.
2. Justify designs by what they make possible or prevent, not by who uses them. Authority
   ("DeepSeek does this") is evidence, not the reason.
3. The reference model is a recurring anchor: when a claim can carry a number (heads of
   width 128, 4096 training tokens, 50 thousand multiply-adds against 805 million), give
   it one. Numbers on the same imaginary model let chapters compare costs without
   introducing new setup.
4. Section headings work best as claims ("Why a table of positions fails", "Two bases in
   one model") rather than topics ("Position tables").

## Math

1. Use display math to compress an idea already stated in words, then unpack it in words
   again after ("Read that back in words…"). The equation is the summary, not the
   exposition.
2. Define notation at first use and reuse it verbatim across chapters ($d$, $d_r$,
   $R(p)$, $\theta$). Symbols with a life of their own, like concatenation $\|$,
   deserve an explicit definition sentence the first time.
3. Inline math for single symbols, display math for derivations. Resist display math for
   things a sentence can carry.
4. A number with a unit inside math carries the unit inside it, as `\text{}`:
   `$= 16.8\,\text{M}$`. Never close the math before the unit (`$= 16.8$M` is a defect),
   and keep `\,` between number and unit.

## Prose texture

1. Short declarative sentences, one claim each. Hedging rarely survives editing here.
2. A few recurring book-level terms do a lot of work: "defect", "ledger", "reference
   model", "what the 2026 models do". Reuse them rather than inventing synonyms.
3. Blockquoted bold asides can forward-reference later chapters ("a chapter 7 preview")
   without derailing the current argument. Use sparingly — one per chapter at most.

## Tables and surveys

1. Model-survey tables work when one column carries the point. Chapter 3's rotation-width
   table is read back with "the fixed value 64 in most rows is the point of the table" —
   if a table needs no interpretation paragraph, ask whether it needs to exist.
2. A "position in recent models" section covers four families in four paragraphs, then
   names the recurring patterns. Per-family paragraphs first, synthesis last.

## Structure habits worth stealing

1. "Where this leads" as a bridge instead of a conclusion.
2. A `Takeaways` section of 10–15 dense numbered items restating each section's
   conclusion, numbered rather than bulleted so a prompt can cite "takeaway 7".
   Expensive to write, cheap to read — skip it if the chapter is short or the takeaways
   would just repeat the roadmap.
3. Prefer numbered lists over bullets throughout chapters, not only in Roadmap and
   Takeaways: any list a reader or prompt might want to point into (survey patterns,
   defect lists, implementation steps) becomes addressable by number.

## Units and numbers

1. Byte and size figures use binary suffixes throughout the book: K = 2^10, M = 2^20,
   G = 2^30, T = 2^40. This is the house convention (colloquial memory sizing); it is
   not the IEC/k8s one — no `i` forms, `KiB` is not used. When converting, recompute
   from the byte count: 5.24 decimal-MB is 5.0 MB, not a relabel.
2. FLOPs and parameter counts keep decimal suffixes (343 GFLOPs, 552B parameters), as do
   hardware throughputs by spec convention (990 TFLOPS, 3.35 TB/s).
3. Never spell out a quantity: "million", "billion", "thousand" become M, B, K.
   Contexts may be written 64K and 1M.
4. A ratio is written with ×: "9.3× fewer", "384× in all". "Times" and "-fold" are
   reserved for counts of occurrences ("runs $S$ times", "used 64 times",
   "reread thousands of times").
5. Every column of a table uses one unit system.

## The reference model

1. Chapters 1 to 6 anchor on the dense 65B model (D = 8192, 80 layers). The attention
   part (5 to 10, 16 to 17) anchors on DSV4.1-Flash: 40 layers, 64 query heads of width
   512, one shared KV latent, contexts of 64K native and 1M via YaRN. State the anchor
   at the chapter's first ledger.
2. Each cross-chapter figure has one owning chapter (chapter 9 owns the pre-selection
   read: 522 MB at 64K, 8.2 G at 1M). Other chapters cite it as "the ledger of
   chapter N" instead of recomputing a variant.
3. Re-deriving a claim on a new anchor can flip it. "Attention is a tenth of the block's
   arithmetic" held on the dense model and fails on the MoE one; re-check imported
   claims before reusing them.

## Notation

1. Positions: $i$ = query, $j$ = key or entry; $t$ only for the decode token in kernel
   chapters. Head $h$. Candidate blocks $b$. The layer index stays in words or as a
   superscript $(\ell)$, never a bare $\ell$.
2. The sink is $\sigma_h$; scores $s_{h,ij}$; the online-softmax running statistics
   $m^{(j)}$ and $\ell$, each defined in words at first use; the selection set
   $\mathcal{S}_i$. Window size $W$ against a layer's window set $\mathcal{W}_\ell$;
   gate logits $\gamma_t$; the pre-pooling candidate $\tilde{c}_t$.

## Algorithms

1. Step-narrated procedures — decode walkthroughs, update rules, loops — go as pseudocode in
   display math; prose carries why (exactness, cost), never what happens in what order. Blocks
   are only as formal as the algorithm needs: some want phase rules and comments, some are five
   lines.
2. The form is a two-column `array{ll}`: statements left, side conditions right ("for each head
   $h$"); a bold signature row names the block, since prose references blocks by name and
   nothing is numbered; italic `//` comments label phases when it helps; `\hline` between
   phases, sparingly; `\quad` indentation. Keywords are `\textbf{for}`/`\textbf{if}`/...,
   assignment is `\leftarrow`, and `=` stays equality.
3. Stick to the command set that renders on both paths (`array`, `\hline`, `\text`/`\textbf`/
   `\textit`, `\operatorname`, `\leftarrow`, `\quad`, `\big[`,`\big]`, `\dots`): KaTeX is the
   tighter side (no `\multicolumn`, no algorithm packages), so anything outside the set gets a
   render check in `just docs/ preview` *and* `just docs/ pdf` before it enters a chapter.
4. Type-check the block as you write it: every line should compose — a $d_c$-wide query does
   not dot a $(d_c{+}d_r)$-wide key. Say a symbol's shape in words whenever the line alone
   doesn't make it obvious.
5. The decode block in `attn/low-rank-attention/mla.md` is the reference example.

## Book and site

1. `docs/evolution/` is the single source of structure for both the website and
   `docs/book.qmd`. A directory = a part or a chapter with subsections; a bare `.md`
   file = a standalone chapter or subsection. Don't hand-list pages in `_quarto.yml`'s
   sidebar — it stays `contents: docs/evolution/**` (auto-discovered from the folder
   tree). A hand-written list silently drifts from the folder tree and from
   `book.qmd`'s include order; it already happened once.
2. Ordering is never encoded in filenames (no `01-`, `02-` prefixes) — use `order:` in
   each page's YAML frontmatter instead, alongside `title:`. This is standard Quarto
   behaviour ("alphabetical by filename unless a numeric `order` field is set"), and it
   means adding or reordering a chapter never requires renaming its neighbours.
   Directory sections take their title/order from that directory's own `index.md`.
3. Adding a new part (e.g. a future FFN or embedding part, alongside the existing
   `attn/`): make a new folder under `docs/evolution/`, give it an `index.md` with
   `title:` and `order:`, and put its chapters inside. No `_quarto.yml` edit needed —
   the sidebar picks it up automatically. Do add its chapters to `docs/book.qmd`'s
   `{{< include ... >}}` list by hand, in the order they should appear in the PDF (that
   list is independent of the sidebar and of directory layout).
4. Every chapter/subsection `.md` file gets `{{< include >}}`-spliced into
   `docs/book.qmd` as raw text, so its YAML frontmatter becomes an extra pandoc
   metadata block merged into the whole book. `title:`/`order:` are safe (inert for
   LaTeX), but never add a PDF-affecting key here (`classoption`, `documentclass`,
   `header-includes`, ...) — it silently changes the whole book, not just that page.
   `docs/justfile`'s `pdf` recipe defends the book's own title with
   `-M title:"mzoo — Evolution"` (CLI metadata always wins over any in-document block).
5. Quarto runs only through docker (`just docs/ render`, `preview`, `pdf`); rendered
   output stays owned by the invoking user.
