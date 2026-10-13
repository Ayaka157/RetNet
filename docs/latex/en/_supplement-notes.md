# `_supplement.tex` — author's notes

Scope: this note records where the English supplement deliberately deviates from the
brief, where it condenses, and what could not be verified. Section numbers quoted below
are the numbers the section receives in the generated English PDF
(`Abstract` is unnumbered, `Introduction` = 1 … `Reference Reliability Note` = 15,
`References` = 16).

## Deliberate deviations

1. **Order.** The supplement is laid out in strict *source* order — Architecture 3, Data 4,
   Training Pipeline 5, Engineering Optimizations 6, Correctness Defects 7, Evaluation 8,
   Limitations 9, Conclusion 10, Appendices A–D 11–14, Reference Reliability Note 15 —
   instead of the brief's literal item order, which placed §7 before §6. Reason: `mktex.py`
   emits Introduction (1) and Background (2) from the source's own English prose
   (`EN_FROM_SOURCE = {"引言", "背景"}`), so source order makes every `\ref` and every
   "Section 6.6"-style cross-reference in the prose resolve to the *original* report
   number. The literal order would have printed §7 before §6 and shifted the rest.
2. **Introduction and Background are not re-authored here.** They come from the source's
   `### 1.2 English` / `### 2.3 English`. (The earlier note in this file's header said the
   same thing; at the time I first read the generated `.tex` that was not yet true —
   `is_english()` then dropped every bilingual title and only `Abstract (English)`
   survived. `mktex.py` was changed to the subsection-level `filter_lang` +
   `EN_FROM_SOURCE` design afterwards, and the claim is now correct. Verified by
   regenerating and reading the `.tex`.)
3. **The reference-reliability note is the last section of the supplement**, immediately
   before `References`, rather than between the Conclusion and Appendix A. It is a note
   *about* the reference list, and the source places it after the reference list too.
4. **§3.4 (chunked recurrence and the closed-form scan) is included in full**, although the
   source also carries a condensed English `### 3.4.3`. The current `mktex.py` emits no
   condensed English subsection at all (only Introduction/Background come from the source),
   so omitting it would leave the English edition with no derivation of the paper's main
   algorithmic contribution. §3.4.1/§3.4.2 exist only in Chinese in the source and are
   translated here; §3.4.3's own wording is reused.
5. **The source's condensed English abstracts are not reproduced as separate subsections.**
   `### 4.4 English (condensed)`, `### 6.9`, `### 9.2` and `### 10.2` are superseded by the
   full §4.1–4.3, §6.1–6.8, §9 and §10 written here. §10 deliberately carries the *full*
   Chinese conclusion content, including the three numbered methodological conclusions,
   which the source's §10.2 only summarises.
6. **§9 contains the extra material the source's §9.2 lacks** (capacity evidence in both
   directions, the memory estimate, the train/inference guard, the unresolved regression and
   the unexplained step-10020 observation). It necessarily restates §9.2's three headline
   limitations, since the sections are merged in the English edition.
7. **Source-title wording.** The source's §7.1 heading reads `_orig_mod.` **后缀**
   (suffix); it is a prefix. The English edition says prefix (both the brief and the
   observed behaviour are unambiguous); this is a deviation from the source heading only,
   not from the source facts.
8. **Appendix A** reproduces the PowerShell block verbatim with only the `#` comments
   translated. **Appendix C** is prose (flag names kept in `\code{}`, descriptions
   translated); the source's four-line Chinese flag block is not reproduced verbatim.

## Condensed, not reproduced

- The source's front-matter blockquote (date / subject / hardware / data / "all numbers on
  AC power") is already typeset by `en/_titles.tex`; it is not repeated.
- The source's §5.1/§5.2 tables are reproduced in full; no rows were dropped.
- No numeric value was rounded, reordered or "tidied". Where the source and the brief
  differ in presentation only (e.g. `3.19 × 4` vs `12.75 ≈ 3.19 × 4`), the source's form is
  kept.

## One rendering deviation in Appendix B

- In snippet **B.4** the two emoji in the `print` strings (`❌` U+274C, `✅` U+2705) are
  emitted through the `listings` escapechar as `|$\times$|` / `|$\checkmark$|`. Consolas has
  no glyph for either codepoint, and the corresponding `literate` entries in
  `preamble-common.tex` do not fire (they are the 24th/25th entries of the list), so both
  characters were being dropped with `Missing character` warnings. The escape renders the
  ×/✓ that `literate` intended. No other byte of the five snippets was changed; the
  remaining Chinese comments are kept verbatim (the preamble supports CJK in `lstlisting`).
- Long unbreakable `\code{}` tokens carry `\allowbreak` (invisible) inside narrow table
  columns and in §7.2's `get_last_checkpoint` sentence, to keep the build free of overfull
  boxes. No identifier text was altered.

## Not independently verified (transcribed from `docs/19` and the brief)

- Every project measurement: step times, throughputs, TFLOPs and shape peaks, kernel
  counts/durations, checkpoint key counts and forward losses, `log_history` lengths,
  benchmark scores/leak rates, and the `training_args.bin` values. I did not re-run
  `scripts/bench_trainer.py`, `scripts/gemm_ceiling.py`, `scripts/profile_kernels.py`,
  `scripts/audit_leak_truncation.py` or any evaluation.
- `docs/18`'s 1.6% claim, `docs/16`'s ~700M-parameter estimate, and the two arXiv
  identifiers (including the suspect `arXiv:2605.16640`) — transcribed and explicitly
  flagged as unverified in §15, as the source does.
- §6.6's `--eval_batches 40` item is carried over as **static review only, not measured**,
  as the source states.

## Build notes (environment, not content)

- `en/_preamble.tex` line 11 previously read
  `\setmonofont{Consolas}[Scale=0.86,Contextuals=Off]`; the installed fontspec rejects `Off`
  and raised 5 fatal errors, so `xelatex` exited 1 with `! LaTeX Error: Key
  'fontspec-opentype/Contextuals' accepts only a fixed set of choices`. The Lead removed the
  option while I was working; the final two passes exit 0. I did not edit that file.
- The same output directory is being rebuilt concurrently by the Lead; one of my in-place
  passes collided with their `xelatex` on the shared `.log`/`.aux` and returned 1. The
  reported result comes from a clean isolated two-pass build of the generated `.tex`
  (`%TEMP%\retnet_enverify`), reproduced in place once the directory was quiet.

## Cosmetic issues outside my two files (for `mktex.py` / the Lead to decide)

Found while verifying the generated `.tex`; none of them come from `_supplement.tex`.

1. **Chinese text still leaks into the English `References` section.** The references body
   is rendered with `render_blocks(refs_body)`, which keeps the source's
   `### 引用可靠性说明` heading and its Chinese bullets, so the English PDF ends with
   `\subsection{引用可靠性说明}` + Chinese prose. It is also a duplicate of the English
   §15 written here. Suggested fix: drop `###` subsections when rendering `refs_body`, or
   run it through the same language filter.
2. **Stray `\subsection{English}` headings** in Introduction (1.1) and Background (2.1):
   `EN_CLEAN` is applied to the top-level name but not to subsection headings. Applying
   `en_clean_heading()` to subsection headings and dropping the ones that clean to the empty
   string would remove them.
3. **Appendix B subsections read "12.1 B.1 Closed-Form State Scan …"**, i.e. the automatic
   number plus the source's own `B.1` label. I kept the `B.1`–`B.5` labels deliberately so
   the English appendix can be cross-referenced against the Chinese edition; strip them from
   my `\subsection{}` titles if the double numbering is unwanted.
