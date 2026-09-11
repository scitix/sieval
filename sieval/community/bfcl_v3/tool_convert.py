"""Vendored verbatim from ShishirPatil/gorilla @ v1.3 (ea13468e).

Upstream path: bfcl_eval/model_handler/utils.py
Upstream blob: f78acc240bee3c43b18de3d4195fbf6ee1a207f1
License: Apache-2.0

Symbols taken (bodies unchanged): `convert_to_tool`, `_cast_to_openai_type`.
A partial extraction -- the rest of `model_handler/utils.py` is out of scope.

Deviations, beyond the usual import rewrite:

- `_cast_to_openai_type` compares its `mapping` argument against the module
  global `GORILLA_TO_OPENAPI`, imported here from the vendored `.type_mappings`.
- `convert_to_tool` branches on upstream's `ModelStyle`
  (`bfcl_eval.model_handler.model_style`). Rather than vendor that file, the
  enum below copies the 11 members these two functions reference, with
  upstream's values. `Gorilla` and `NEXUS` are omitted: present upstream, read
  by neither copied function.
"""

import copy
import json
import re
from enum import Enum

from .type_mappings import GORILLA_TO_OPENAPI


class ModelStyle(Enum):
    OpenAI_Completions = "openai-completions"
    OpenAI_Responses = "openai-responses"
    Anthropic = "claude"
    Mistral = "mistral"
    GOOGLE = "google"
    AMAZON = "amazon"
    FIREWORK_AI = "firework_ai"
    OSSMODEL = "ossmodel"
    COHERE = "cohere"
    WRITER = "writer"
    NOVITA_AI = "novita_ai"


def _cast_to_openai_type(properties, mapping):
    for key, value in properties.items():
        if "type" not in value:
            properties[key]["type"] = "string"
        else:
            var_type = value["type"]
            if mapping == GORILLA_TO_OPENAPI and var_type == "float":
                properties[key]["format"] = "float"
                properties[key]["description"] += " This is a float type value."
            if var_type in mapping:
                properties[key]["type"] = mapping[var_type]
            else:
                properties[key]["type"] = "string"

        # Currently support:
        # - list of any
        # - list of list of any
        # - list of dict
        # - list of list of dict
        # - dict of any

        if properties[key]["type"] == "array" or properties[key]["type"] == "object":
            if "properties" in properties[key]:
                properties[key]["properties"] = _cast_to_openai_type(
                    properties[key]["properties"], mapping
                )
            elif "items" in properties[key]:
                properties[key]["items"]["type"] = mapping[properties[key]["items"]["type"]]
                if (
                    properties[key]["items"]["type"] == "array"
                    and "items" in properties[key]["items"]
                ):
                    properties[key]["items"]["items"]["type"] = mapping[
                        properties[key]["items"]["items"]["type"]
                    ]
                elif (
                    properties[key]["items"]["type"] == "object"
                    and "properties" in properties[key]["items"]
                ):
                    properties[key]["items"]["properties"] = _cast_to_openai_type(
                        properties[key]["items"]["properties"], mapping
                    )
    return properties


def convert_to_tool(functions, mapping, model_style):
    functions = copy.deepcopy(functions)
    oai_tool = []
    for item in functions:
        if "." in item["name"] and model_style in [
            ModelStyle.OpenAI_Completions,
            ModelStyle.OpenAI_Responses,
            ModelStyle.Mistral,
            ModelStyle.GOOGLE,
            ModelStyle.OSSMODEL,
            ModelStyle.Anthropic,
            ModelStyle.COHERE,
            ModelStyle.AMAZON,
            ModelStyle.NOVITA_AI,
        ]:
            # OAI does not support "." in the function name so we replace it with "_". ^[a-zA-Z0-9_-]{1,64}$ is the regex for the name.
            item["name"] = re.sub(r"\.", "_", item["name"])

        item["parameters"]["type"] = "object"
        item["parameters"]["properties"] = _cast_to_openai_type(
            item["parameters"]["properties"], mapping
        )

        if model_style == ModelStyle.Anthropic:
            item["input_schema"] = item["parameters"]
            del item["parameters"]

        if model_style == ModelStyle.AMAZON:
            item["inputSchema"] = {"json": item["parameters"]}
            del item["parameters"]

        if model_style in [
            ModelStyle.GOOGLE,
            ModelStyle.WRITER,
        ]:
            # Remove fields that are not supported by Gemini or Palmyra.
            # No `optional` field in function schema.
            if "optional" in item["parameters"]:
                del item["parameters"]["optional"]
            for params in item["parameters"]["properties"].values():
                if "description" not in params:
                    params["description"] = ""
                # No `default` field in GOOGLE or Palmyra's schema.
                if "default" in params:
                    params["description"] += f" Default is: {str(params['default'])}."
                    del params["default"]
                # No `optional` field in parameter schema as well.
                if "optional" in params:
                    params["description"] += f" Optional: {str(params['optional'])}."
                    del params["optional"]
                # No `required` field in parameter schema as well.
                if "required" in params:
                    params["description"] += f" Required: {str(params['required'])}."
                    del params["required"]
                # No `maximum` field.
                if "maximum" in params:
                    params["description"] += f" Maximum value: {str(params['maximum'])}."
                    del params["maximum"]
                # No `minItems` field.
                if "minItems" in params:
                    params[
                        "description"
                    ] += f" Minimum number of items: {str(params['minItems'])}."
                    del params["minItems"]
                # No `maxItems` field.
                if "maxItems" in params:
                    params[
                        "description"
                    ] += f" Maximum number of items: {str(params['maxItems'])}."
                    del params["maxItems"]
                # No `additionalProperties` field.
                if "additionalProperties" in params:
                    params[
                        "description"
                    ] += f" Additional properties: {str(params['additionalProperties'])}."
                    del params["additionalProperties"]
                # For Gemini, only `enum` field when the type is `string`.
                # For Palmyra, `enum` field is not supported.
                if "enum" in params and (
                    model_style == ModelStyle.WRITER
                    or (model_style == ModelStyle.GOOGLE and params["type"] != "string")
                ):
                    params["description"] += f" Enum values: {str(params['enum'])}."
                    del params["enum"]
                # No `format` when type is `string`
                if "format" in params and params["type"] == "string":
                    params["description"] += f" Format: {str(params['format'])}."
                    del params["format"]

        # Process the return field
        if "response" in item:
            if model_style in [
                ModelStyle.Anthropic,
                ModelStyle.GOOGLE,
                ModelStyle.FIREWORK_AI,
                ModelStyle.WRITER,
                ModelStyle.AMAZON,
                ModelStyle.NOVITA_AI,
            ]:
                item[
                    "description"
                ] += f" The response field has the following schema: {json.dumps(item['response'])}"
                del item["response"]

        if model_style in [
            ModelStyle.Anthropic,
            ModelStyle.GOOGLE,
            ModelStyle.OSSMODEL,
        ]:
            oai_tool.append(item)
        elif model_style in [
            ModelStyle.OpenAI_Responses
        ]:
            oai_tool.append({"type": "function", 
                             "name": item["name"], 
                             "description": item["description"], 
                             "parameters": item["parameters"]})
        elif model_style in [
            ModelStyle.COHERE,
            ModelStyle.OpenAI_Completions,
            ModelStyle.Mistral,
            ModelStyle.FIREWORK_AI,
            ModelStyle.WRITER,
            ModelStyle.NOVITA_AI,
        ]:
            oai_tool.append({"type": "function", "function": item})
        elif model_style == ModelStyle.AMAZON:
            oai_tool.append({"toolSpec": item})

    return oai_tool
