"""AllenAI IFBench evaluation adaptation.

Source: https://github.com/allenai/IFBench
Revision: 1091c4c3de6c1f6ed12c012ed68f11ea450b0117

Local adaptations:
- Convert same-directory imports to package-relative imports.
- Store NLTK data under SIEVAL_IFBENCH_NLTK_DATA or a user cache directory,
  and register that path through NLTK_DATA/nltk.data.path so evaluator imports
  do not write generated data into the source tree.
- Give both graders in evaluation_lib.py a keyword-only `instruction_dict`,
  defaulting to the vendored registry, so a caller can grade through a repaired
  overlay without mutating a module-level global that concurrently-graded
  samples share. Omitting it reproduces upstream exactly. Only the graders'
  signatures change; instructions.py and instructions_registry.py are
  untouched, so the checkers themselves stay byte-identical to upstream.

AI-Generated Code - GPT-5 (OpenAI)
"""
