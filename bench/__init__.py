"""RetNet capability benchmark v3 — declarative item battery.

See ``bench/CONTRACT.md`` for the item schema and the checker DSL, and
``docs/14_基准测试v3.md`` for the design rationale and measured results.

Layout::

    bench/
      CONTRACT.md        authoring contract (schema + DSL reference)
      checkers.py        checker-DSL compiler -> closures
      registry.py        item dataclass, validation, id registry
      gen_basics.py      deterministic 填空/问答/语言/续写 generator
      items/*.jsonl      authored items (one JSON object per line)
      pools/*.json      raw content packs consumed by gen_basics
      data/              frozen, hashed dataset produced by compile_bench
"""
