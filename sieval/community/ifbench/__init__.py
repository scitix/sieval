"""AllenAI IFBench evaluation adaptation.

Source: https://github.com/allenai/IFBench
Revision: 1091c4c3de6c1f6ed12c012ed68f11ea450b0117

Local adaptations:
- Convert same-directory imports to package-relative imports.
- Store NLTK data under SIEVAL_IFBENCH_NLTK_DATA or a user cache directory,
  and register that path through NLTK_DATA/nltk.data.path so evaluator imports
  do not write generated data into the source tree.

AI-Generated Code - GPT-5 (OpenAI)
"""
