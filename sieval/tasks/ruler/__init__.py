"""RULER generative tasks — long-context benchmark (4 categories, 13 configs).

Concrete tasks are lazy-loaded by the top-level ``sieval.tasks`` package; this
module is intentionally import-light. The 13 RULER configs are produced from 5
parameterized (Dataset, Task) pairs via YAML ``args``. VT and CWE embed one
in-context demonstration (mirroring upstream RULER), so they are ``kshot``
(``n_shot=1``); the rest are genuinely 0-shot:

  - Retrieval / NIAH  → ruler_niah_0shot_gen (8 configs: single_1/2/3,
    multikey_1/2/3, multivalue, multiquery)
  - Multi-hop tracing → ruler_vt_kshot_gen  (vt)
  - Aggregation       → ruler_cwe_kshot_gen (cwe), ruler_fwe_0shot_gen (fwe)
  - QA                → ruler_qa_0shot_gen  (2 configs: squad, hotpotqa)

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""
