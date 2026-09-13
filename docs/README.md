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
