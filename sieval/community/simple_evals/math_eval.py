# https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/math_eval.py

# Original QUERY_TEMPLATE from OpenAI simple-evals (DO NOT MODIFY - used by existing tasks)
QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

# 4-shot examples for MATH evaluation (for few-shot tasks)
MATH_FEW_SHOT_EXAMPLES = [
    {
        "problem": "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{align*} 30n&=480\\\\ \\Rightarrow\\qquad n&=480/30=\\boxed{16} \\end{align*}",
    },
    {
        "problem": "If the system of equations \\begin{align*} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$",
    },
]

# Additional templates for few-shot evaluation
FEW_SHOT_TEMPLATE = """Problem: {problem}
Solution: {solution}
"""

FEW_SHOT_QUERY_TEMPLATE = """{few_shot_str}
Problem: {problem}
Solution: """
