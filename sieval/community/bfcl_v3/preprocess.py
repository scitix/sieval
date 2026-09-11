"""Vendored verbatim from ShishirPatil/gorilla @ v1.3 (ea13468e).

Upstream path: bfcl_eval/model_handler/utils.py
Upstream blob: f78acc240bee3c43b18de3d4195fbf6ee1a207f1
License: Apache-2.0

Symbols taken (bodies unchanged): `system_prompt_pre_processing_chat_model`,
`_get_language_specific_hint`, `func_doc_language_specific_pre_processing`.
This is a partial extraction, not a whole-file copy -- the rest of
`bfcl_eval/model_handler/utils.py` (multi-turn state helpers, response
formatters, the other model families' prompt builders) is out of scope.

Both public symbols run on the way OUT, shaping what the model is shown. They
are deliberately not applied on the way back in: upstream's `ast_file_runner`
grades against the *unprocessed* schema, so the checker still sees the real
declared types.

Only deviation: upstream imports `DEFAULT_SYSTEM_PROMPT` from
`bfcl_eval.constants.default_prompts`; that is rewritten to the relative import
of this package's copy.
"""

import json

from .prompts import DEFAULT_SYSTEM_PROMPT


def system_prompt_pre_processing_chat_model(prompts, function_docs, test_category):
    """
    Add a system prompt to the chat model to instruct the model on the available functions and the expected response format.
    If the prompts list already contains a system prompt, append the additional system prompt content to the existing system prompt.
    """
    assert type(prompts) == list

    system_prompt_template = DEFAULT_SYSTEM_PROMPT

    system_prompt = system_prompt_template.format(functions=function_docs)

    # System prompt must be in the first position
    # If the question comes with a system prompt, append its content at the end of the chat template.
    if prompts[0]["role"] == "system":
        prompts[0]["content"] = system_prompt + "\n\n" + prompts[0]["content"]
    # Otherwise, use the system prompt template to create a new system prompt.
    else:
        prompts.insert(
            0,
            {"role": "system", "content": system_prompt},
        )

    return prompts


def _get_language_specific_hint(test_category):
    if test_category == "java":
        return " Note that the provided function is in Java 8 SDK syntax."
    elif test_category == "javascript":
        return " Note that the provided function is in JavaScript syntax."
    else:
        return " Note that the provided function is in Python 3 syntax."


def func_doc_language_specific_pre_processing(function, test_category):
    if len(function) == 0:
        return function

    assert type(function) == list
    for item in function:
        # Add language specific hints to the function description
        func_description = item["description"]
        item["description"] = item["description"] + _get_language_specific_hint(
            test_category
        )
        # Process the parameters
        properties = item["parameters"]["properties"]
        if test_category == "java":
            for key, value in properties.items():
                if value["type"] == "any":
                    properties[key][
                        "description"
                    ] += " This parameter can be of any type of Java object in string representation."
                else:
                    value[
                        "description"
                    ] += f" This is Java {value['type']} type parameter in string representation."
                if value["type"] == "ArrayList" or value["type"] == "Array":
                    value[
                        "description"
                    ] += f" The list elements are of type {value['items']['type']}; they are not in string representation."
                    del value["items"]

                value["type"] = "string"

        elif test_category == "javascript":
            for key, value in properties.items():
                if value["type"] == "any":
                    properties[key][
                        "description"
                    ] += " This parameter can be of any type of JavaScript object in string representation."
                else:
                    value[
                        "description"
                    ] += f" This is JavaScript {value['type']} type parameter in string representation."
                if value["type"] == "array":
                    value[
                        "description"
                    ] += f" The list elements are of type {value['items']['type']}; they are not in string representation."
                    del value["items"]

                if value["type"] == "dict":
                    if "properties" in value:  # not every dict has properties
                        value[
                            "description"
                        ] += f" The dictionary entries have the following schema; they are not in string representation. {json.dumps(value['properties'])}"
                        del value["properties"]

                value["type"] = "string"

    return function
