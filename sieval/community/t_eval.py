import ast
import json
import re
from dataclasses import dataclass
from string import Formatter
from typing import Any

# to fill the empty string for embeddings
EMB_PLACEHOLDER = "###EMPTYSTRING###"


# adapted from https://github.com/open-compass/T-Eval/blob/58f22406404d7e2a4f36856a19c7f4dc28a0a5f0/teval/schema.py
@dataclass
class ResponseDataSample:
    """
    Args:
        template(str): Format string with keyword-only arguments. For
            example '{who} like {what}'
        pred(Any): Parsed data from LLM generating response.
        gt(Any): Ground truth data
        meta_data(dict, optional): Meta information will be used to evaluate
             LLM's response
    """

    template: str
    pred: Any
    gt: Any
    meta_data: dict = None


# adapted from https://github.com/open-compass/T-Eval/blob/58f22406404d7e2a4f36856a19c7f4dc28a0a5f0/teval/utils/format_load.py
def format_load(raw_data: str, start_character: str = "", end_character: str = ""):
    """Format the raw data into the format that can be evaluated.

    Args:
        raw_data (str): The raw data.
        start_character (str, optional): The start character. Defaults to '', if using it, the string will be sliced from the first start_character.
        end_character (str, optional): The end character. Defaults to '', if using it, the string will be sliced to the last end_character.

    Returns:
        str: The formatted data.
    """
    if not isinstance(raw_data, str):
        # the data has been evaluated
        return raw_data
    if "```json" in raw_data:
        raw_data = raw_data[raw_data.find("```json") + len("```json") :]
        raw_data = raw_data.strip("`")
    if start_character != "":
        raw_data = raw_data[raw_data.find(start_character) :]
    if end_character != "":
        raw_data = raw_data[: raw_data.rfind(end_character) + len(end_character)]
    successful_parse = False
    try:
        data = ast.literal_eval(raw_data)
        successful_parse = True
    except Exception:
        pass
    try:
        if not successful_parse:
            data = json.loads(raw_data)
        successful_parse = True
    except Exception:
        pass
    try:
        if not successful_parse:
            data = json.loads(raw_data.replace("'", '"'))
        successful_parse = True
    except Exception:
        pass
    if not successful_parse:
        raise Exception("Cannot parse raw data")
    return data


# adapted from https://github.com/open-compass/T-Eval/blob/58f22406404d7e2a4f36856a19c7f4dc28a0a5f0/teval/utils/template.py
def format_string(template: str, input_data: dict) -> str:
    """Return string with input content according input format template.

    Args:
        template (str): Format string with keyword-only argument. For
            example '{who} like {what}'
        input_data (dict): Input data to fill in the input template.

    Returns:
        str: Return string.
    """

    return template.format(**input_data)


def parse_string(
    template: str, input_string: str, allow_newline: bool = False
) -> dict | None:
    """Return a dictionary whose keys are from input template and value is
    responding content from input_string.

    Args:
        template (str): Format template with keyword-only argument. For
            example '{who} like {what}'
        input_string (str): Input string will be parsed.
        allow_newline (boolen): Whether allow '\n' in {} during RE match, default to False.

    Returns:
        dict: Parsed data from input string according to format string. If
            input string doesn't match template, It will return None.

    Examples:
        >>> template = '{who} like {what}'
        >>> input_string = 'monkey like banana'
        >>> data = parse_string(template, input_string)
        >>> data
        >>> {'who': 'monkey', 'what': 'banana'}
        >>> input_string = 'monkey likes banana'
        >>> data = parse_string(template, input_string)
        >>> data
        >>> None
        >>> template = '{what} like {what}'
        >>> input_string = 'monkey like banana'
        >>> data = parse_string(template, input_string)
        >>> data
        >>> {'what': ['monkey', 'banana']}
    """

    formatter = Formatter()
    context = []
    keys = []
    for v in formatter.parse(template):
        # v is (literal_text, field_name, format_spec, conversion)
        if v[1] is not None:
            keys.append(v[1])
        context.append(v[0])
    pattern = template
    for k in keys:
        pattern = pattern.replace("{" + f"{k}" + "}", "(.*)")
    # pattern = re.compile(rf'{pattern}')
    values = re.findall(pattern, input_string, re.S if allow_newline else 0)
    if len(values) < 1:
        return None
    data = dict()
    for k, v in zip(keys, values[0]):
        if k in data:
            tmp = data[k]
            if isinstance(tmp, list):
                data[k].append(v)
            else:
                data[k] = [tmp, v]
        else:
            data[k] = v
    return data
