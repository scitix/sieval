# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vendored from facebookresearch/Multi-IF (Apache-2.0), file `metrics.py`:
# https://github.com/facebookresearch/Multi-IF/blob/1cdb53ed18499ad729e0766e5d3099dd5344406f/metrics.py
#
# Only the two per-response graders are taken. Upstream's surrounding
# `MultiTurnInstructionFollowingPromptSolution` drives a pandas/scipy pipeline
# that re-reads a CSV per turn and bootstraps confidence intervals; SiEval's
# task owns conversation assembly and aggregation instead, so importing that
# machinery would pull pandas + scipy in for two pure functions.
#
# Local adaptations:
#   1. `import ifeval` -> `from . import ifeval` (upstream is a flat repo).
#   2. The `Dict[str, float]` return annotations are corrected to `dict`: both
#      functions return lists, not floats, so upstream's annotation is wrong.
#   3. Both graders take a keyword-only `instruction_dict`, defaulting to the
#      vendored registry, so `multi_if_0shot_gen_fixed` can grade through the
#      repaired checkers in `sieval.community.instruction_following_eval_fixed`
#      without mutating a global that concurrently-graded samples share.
#      Omitting the argument reproduces upstream's behaviour exactly, which is
#      what the unqualified task does.
# Otherwise the bodies are byte-identical to upstream.

from typing import Any

from . import ifeval


def gen_acc_strict(x: dict[str, Any], *, instruction_dict=None) -> dict:
    # reference: fbcode/gen_ai/github/fair_evals/evals/tasks/finetune/ifeval.py
    if instruction_dict is None:
        instruction_dict = ifeval.INSTRUCTION_DICT
    response = str(x["response"])
    instruction_list = x["instruction_id_list"]
    is_following_list = []
    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instruction_dict[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**x["kwargs"][index])

        if response and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return {
        "follow_instruction_list": is_following_list,
        "instruction_id_list": instruction_list,
    }


def gen_acc_loose(x: dict[str, Any], *, instruction_dict=None) -> dict:
    if instruction_dict is None:
        instruction_dict = ifeval.INSTRUCTION_DICT
    response = str(x["response"])
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = x["instruction_id_list"]
    is_following_list = []
    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instruction_dict[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**x["kwargs"][index])

        is_following = False
        for r in all_responses:  # type: ignore
            if r.strip() and instruction.check_following(r):  # type: ignore
                is_following = True
                break

        is_following_list.append(is_following)
    return {
        "follow_instruction_list": is_following_list,
        "instruction_id_list": instruction_list,
    }
