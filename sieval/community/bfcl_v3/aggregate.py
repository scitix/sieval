"""Vendored verbatim from ShishirPatil/gorilla @ v1.3 (ea13468e).

Upstream path: bfcl_eval/eval_checker/eval_runner_helper.py
Upstream blob: 1de6462be8be9cc91d5a0e95c69e2c6fd6fc1a0d
License: Apache-2.0

Symbols taken (bodies unchanged): `calculate_weighted_accuracy`,
`calculate_unweighted_accuracy`. This is a partial extraction, not a
whole-file copy -- the rest of `eval_runner_helper.py` (leaderboard CSV
generation, cost/latency aggregation, pandas/numpy-based reporting) is out of
scope for this port.

Only deviation: neither function needs any import, so there was nothing to
rewrite.
"""


def calculate_weighted_accuracy(accuracy_dict_list, display_na_if_category_missing=True):
    has_na = False
    total_count = 0
    total_accuracy = 0
    for accuracy_dict in accuracy_dict_list:
        accuracy = accuracy_dict["accuracy"]
        count = accuracy_dict["total_count"]
        if accuracy_dict["display_accuracy"] == "N/A":
            has_na = True

        total_count += count
        total_accuracy += accuracy * count

    result = {"accuracy": total_accuracy / total_count, "total_count": total_count}

    if has_na and display_na_if_category_missing:
        result["display_accuracy"] = "N/A"
    else:
        result["display_accuracy"] = result["accuracy"]

    return result


def calculate_unweighted_accuracy(accuracy_dict_list, display_na_if_category_missing=True):
    has_na = False
    total_count = 0
    total_accuracy = 0
    for accuracy_dict in accuracy_dict_list:
        accuracy = accuracy_dict["accuracy"]
        count = accuracy_dict["total_count"]
        if accuracy_dict["display_accuracy"] == "N/A":
            # If a category is not being evaluated, it will still be considered 0 in the overall score calculation.
            has_na = True

        total_count += count
        total_accuracy += accuracy

    result = {
        "accuracy": total_accuracy / len(accuracy_dict_list),
        "total_count": total_count,
    }

    if has_na and display_na_if_category_missing:
        result["display_accuracy"] = "N/A"
    else:
        result["display_accuracy"] = result["accuracy"]

    return result
