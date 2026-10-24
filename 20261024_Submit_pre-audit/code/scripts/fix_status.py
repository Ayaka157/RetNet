"""Single source of truth for fix status (docs/00F1, docs/01, README).

## Why this file exists (P0-5)

The submission had three mutually inconsistent status statements: the README
said all 27 items had landed, `docs/00F1` said "fixed" per item, while the
`docs/01` table marked E10/E11/E12/E15/E18/E19/E21/E22/E23 as "not fixed" and
E07 as "partially fixed". `verify.py` only checked byte-identical code blocks
and **never looked at the status table**, so "✅ all checks passed" could
coexist with a table saying nine fixes were missing.

Now the generators and the checker import this dict, and `verify.py` asserts
that the docs/01 table, docs/00F1 `**Status**` lines and the README table all
equal it.

Change a status here only, then regenerate:

    python scripts/_gen_code_doc.py
    python scripts/_gen_fix_doc.py
"""
from __future__ import annotations

#: id -> status. These exact strings are written into all three documents;
#: verify.py requires all four copies (this one included) to match.
FIX_STATUS: dict[str, str] = {
    "E01": "Fixed",
    "E02": "Fixed",
    "E03": "Fixed",
    "E04": "Fixed",
    "E05": "Fixed",
    "E06": "Fixed",
    "E07": "Fixed",
    "E08": "Fixed",
    "E09": "Fixed",
    "E10": "Fixed",
    "E11": "Fixed",
    "E12": "Fixed",
    "E13": "Fixed",
    "E14": "Fixed",
    "E15": "Fixed",
    "E16": "Fixed",
    "E17": "Fixed",
    "E18": "Fixed",
    "E19": "Fixed",
    "E20": "Fixed",
    "E21": "Fixed",
    "E22": "Fixed",
    "E23": "Fixed",
    "N01": "Fixed",
    "N02": "Doc corrected (see N01)",
    "N03": "Doc corrected (no code)",
    "N04": "Fixed",
}

#: Short "where" column shared by docs/01 and the README table.
FIX_NOTE: dict[str, str] = {
    "E01": "`grow_500m.py` `fill()`: new columns zero, new rows random",
    "E02": "`grow_500m.py` `grow_norm()`: blocks 1.0, final 0.0 (whole segment)",
    "E03": "`grow_500m.py` embedding `[grow_old_dim:]` re-randomised",
    "E04": "`grow_500m.py`: 5-step backward + refuse to save on zero grads",
    "E05": "`GrowthLayerNorm` default `gamma_new=1.0`",
    "E06": "`train.py`: default on + `--decay_1d` + optimizer-group assertion",
    "E07": "`GradCoverageCallback`: zero-grad check after step 1",
    "E08": "`_preflight` live ratio + `--allow_dead_params` + baseline",
    "E09": "`build_flat_corpus.py`: `--add-eos` defaults to True",
    "E10": "`_data_supervisor.py` / `_prep_ext.py` pass `--add-eos`",
    "E11": "`build_ctx_mix.py` refuses sources with `add_eos=false`",
    "E12": "`Stage.require_eos` end-to-end + `_FlatStage` sampling",
    "E13": "producer writes `rebuild_cmd`/`source_ids`; `gc_stage` 3 checks",
    "E14": "`assert_val_disjoint`: path + declared source + row hashes",
    "E15": "`load_best_model_at_end=False`; offline selection (docs/02)",
    "E16": "removed the two parallel-only normalisations",
    "E17": "`retnet_hf/defaults.py` single constant 4",
    "E18": "`should_accept_reschedule` rejects LR jumps by default",
    "E19": "`_FlatStage.refresh` compares scan-time signatures",
    "E20": "`run_task` plan branch passes `eval_batch_mult`",
    "E21": "recurrent `tgt_len==1` fast path",
    "E22": "`_chunked_weighted_ce` chunks by byte budget",
    "E23": "every fix has a test that fails when reverted",
    "N01": "`reinit_gamma.py` per-head, plus shard merge",
    "N02": "reporting: all-head and live-head medians",
    "N03": "layer width from per-tensor live rows/columns",
    "N04": "`eval_holdout.py` pinned to chunkwise",
}
