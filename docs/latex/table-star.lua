-- Promote pandoc tables to full-width floats for twocolumn LaTeX.
-- pandoc emits longtable, which LaTeX rejects in twocolumn mode; this filter
-- rewrites each table's LaTeX fragment into \begin{table*}...\end{table*}
-- with a plain tabular. Pure string transform, no interpretation.
local function promote(tbl)
  local doc = pandoc.Pandoc({tbl})
  -- columns=1000 keeps the column spec on one line for the pattern below
  local frag = pandoc.write(doc, {format = "latex", columns = 1000})
  local caption = frag:match("\\caption%s*(%b{})") or ""
  frag = frag:gsub("\\caption%s*%b{}%s*\\tabularnewline%s*", "")
  -- equal-width columns across the full page (needs array package for >{})
  local n = math.max(#(tbl.colspec or {}), 1)
  local col = ">{\\raggedright\\arraybackslash}p{\\dimexpr \\textwidth/"
    .. n .. " - 2\\tabcolsep\\relax}"
  local spec = string.rep(col, n)
  -- drop longtable's repeated header/footer machinery
  frag = frag:gsub("\\endfirsthead.-\\endhead%s*", "")
  frag = frag:gsub("\\endfirsthead%s*", "")
  frag = frag:gsub("\\endhead%s*", "")
  frag = frag:gsub("\\endfoot%s*", "")
  frag = frag:gsub("\\bottomrule%s*\\noalign{}%s*\\endlastfoot%s*", "")
  frag = frag:gsub("\\endlastfoot%s*", "")
  local body = frag:match("\n(.-)%s*\\end{longtable}") or ""
  local out = {"\\begin{table*}[t]"}
  out[#out + 1] = "\\centering\\small"
  if caption ~= "" then
    out[#out] = out[#out] .. "\\caption" .. caption
  end
  out[#out + 1] = "\\begin{tabular}" .. spec:gsub("@{}", "")
  out[#out + 1] = body
  out[#out + 1] = "\\bottomrule"
  out[#out + 1] = "\\end{tabular}\\end{table*}"
  return pandoc.RawBlock("latex", table.concat(out, "\n"))
end
return {{Table = promote}}
