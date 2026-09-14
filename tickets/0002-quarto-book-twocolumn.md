# quarto whole-book PDF: unexplained `twocolumn` injection breaks longtable

State when this ticket was filed (commit `19b4bf2`):

- quarto 1.7.33 running inside docker image `mzoo-quarto` (built from `docs/docker/Dockerfile`:
  ubuntu 24.04 arm64 + quarto 1.7.33 tarball + `texlive-xetex texlive-fonts-recommended
  texlive-plain-generic lmodern`)
- `docs/book.qmd`: one qmd that pulls all 40 chapters in via `{{< include ... >}}`, pdf format:
  documentclass article, papersize a4, geometry margin=2.2cm, fontsize 10pt, colorlinks, toc,
  toc-depth 2, number-sections false, citeproc, header-includes calc+array, bibliography
  `references.bib` (top-level: `link-citations: true`)
- rendered from repo root: `quarto render docs/book.qmd --to pdf --output <f>` (repo root has a
  website project `_quarto.yml`, format block is html-only)
- repro: `just docs/ pdf`

## problem

LaTeX compile fails at the first longtable (the reference-model table in `docs/evolution/index.md`):

```
! Missing number, treated as zero.
<to be read again> (
l.232 \begin{minipage}[b]{\linewidth}\raggedright
```

Generated `book.tex` line 227+ is stock pandoc longtable output: `p{(\linewidth - 4\tabcolsep) *
\real{0.3333}}` column specs and `\begin{minipage}[b]{\linewidth}` header cells.

## observed in generated tex

`book.tex` preamble has:

```latex
\documentclass[
  10pt,
  a4paper,
  twocolumn]{article}
...
\usepackage{supertabular}
\let\longtable\supertabular
\let\endlongtable\endsupertabular
```

(the supertabular block appears **twice**)

The `\let\longtable\supertabular` swap is what breaks the compile: supertabular cannot parse
pandoc's longtable column spec (`\real{...}` / `(\linewidth - 4\tabcolsep)`), producing the "Missing
number" error at the first header minipage.

`twocolumn` is the enabler: quarto only swaps longtable→supertabular in two-column layouts.

## findings

1. Nothing in the repo asks for twocolumn:
   - `grep twocolumn` over `_quarto.yml`, `docs/book.qmd`, all `docs/**/*.md`, `docs/latex/*` finds
     no metadata source (only `docs/latex/table-star.lua` *comments* mention twocolumn; that lua
     filter is not referenced by the quarto render)
2. Not in quarto itself:
   - `grep -c twocolumn /opt/quarto-1.7.33/bin/quarto.js` → 0
   - `grep -rln twocolumn /opt/quarto-1.7.33/share/` → nothing
   - i.e. no quarto template or pandoc default latex template in the install contains the string
3. Same content compiles standalone: a one-chapter qmd (`{{< include evolution/index.md >}}` with
   the same pdf format keys) produced `_one.tex` with `\documentclass[10pt,a4paper,]{article}`, no
   twocolumn, no supertabular, and rendered to pdf successfully.
4. Removing any single format key from book.qmd's yaml one at a time (documentclass, geometry,
   colorlinks, citeproc, header-includes, toc, toc-depth) — twocolumn still present in every
   variant, and without `documentclass: article` the class defaults to `scrartcl` (still with
   twocolumn).
5. Both the failing book.qmd and the succeeding one-chapter test were rendered the same way, from
   the same cwd (repo root, inside the website project context). The only structural differences
   between the two inputs: number/specifics of includes and the exact yaml key set. The decisive
   trigger was not isolated.
6. Notably, `scrartcl` appearing when `documentclass` is removed is itself suspicious — pandoc's
   default documentclass is `article`, so something is overriding/transforming documentclass and
   class options for this file.

## version info

- quarto 1.7.33 (linux arm64 tarball)
- texlive from ubuntu 24.04 (texlive-xetex etc.)
- pandoc: whatever quarto 1.7.33 bundles
