# adapted from https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/humaneval_eval.py
QUERY_TEMPLATE = """
Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
{prompt}
""".strip()
