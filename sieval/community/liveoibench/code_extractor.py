# adapted from https://github.com/LiveOIBench/LiveOIBench-Evaluation/blob/7759e3b8672307cfbdc8ab8e679bd87cc1dd4c12/src/code_extractor.py
"""
Utility for extracting code from LLM responses.

This module provides a centralized way to extract and save code from
various LLM response formats.
"""

import re
import os
import json
from typing import Optional, Tuple


class CodeExtractor:
    """Utility class for extracting code from LLM responses."""

    @staticmethod
    def extract_code(content: str, task_name: str) -> Tuple[Optional[str], str]:
        """
        Extract code from LLM response content using prioritized patterns.

        Args:
            content: The response content from the LLM
            task_name: Name of the task (used for pattern matching)

        Returns:
            Tuple of (extracted_code, status) where status is one of:
            'success', 'empty', 'not_found'
        """
        if not content:
            return None, 'not_found'

        # Prioritized patterns for code extraction
        patterns = [
            # 1) ```{task}.cpp or ```{task}.c ... ```
            rf"```(?:{re.escape(task_name)}\.(?:cpp|c))\s*([\s\S]*?)```",
            # 2) ```cpp or ```c ... ```
            r"```(?:cpp|c)\s*([\s\S]*?)```",
            # 3) Generic ``` ... ```
            r"```([\s\S]*?)```",
        ]

        code = None
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                # Get the longest code block
                code = max(matches, key=lambda x: len(x.split('\n'))).strip()
                break

        if code is None:
            return None, 'not_found'

        if not code:
            return "", 'empty'

        # Post-process the code
        code = CodeExtractor._post_process_code(code)
        return code, 'success'

    @staticmethod
    def _post_process_code(code: str) -> str:
        """
        Post-process extracted code to fix common issues.

        Args:
            code: Raw extracted code

        Returns:
            Cleaned code
        """
        lines = code.split('\n')

        # Check if the code is too short
        if len(lines) < 5:
            # This might be an issue but we'll keep it
            pass

        # Fix first line if it contains just "cpp" or ends with .cpp
        if lines:
            first_line = lines[0].strip()
            if first_line.lower() == "cpp" or first_line.endswith(".cpp"):
                lines[0] = "// " + first_line

        return '\n'.join(lines)

    @staticmethod
    def save_code(
        code: Optional[str],
        prediction_path: str,
        task: str,
        model: str,
        seed: int
    ):
        """
        Save extracted code to a file.

        Args:
            code: Extracted code (can be None or empty)
            prediction_path: Path to save predictions
            task: Task name
            model: Model name
            seed: Random seed number
        """
        # Create codes directory if it doesn't exist
        os.makedirs(os.path.join(prediction_path, "codes"), exist_ok=True)

        # Save code (even if empty)
        code_path = os.path.join(prediction_path, "codes", f"{task}_{model}_{seed}.cpp")
        with open(code_path, "w") as f:
            f.write(code if code else "")

    @staticmethod
    def save_raw_response(
        response: any,
        prediction_path: str,
        task: str,
        model: str,
        seed: int
    ):
        """
        Save raw API response to a JSON file.

        Args:
            response: API response object
            prediction_path: Path to save predictions
            task: Task name
            model: Model name
            seed: Random seed number
        """
        if response is None:
            return

        # Skip empty responses
        if hasattr(response, "choices") and hasattr(response, "usage"):
            if response.usage.completion_tokens == 0:
                return

        # Create raw directory if it doesn't exist
        os.makedirs(os.path.join(prediction_path, "raw"), exist_ok=True)

        # Save raw response
        raw_path = os.path.join(prediction_path, "raw", f"{task}_{model}_{seed}.json")

        # Handle different response formats
        try:
            if hasattr(response, "model_dump_json"):
                with open(raw_path, "w") as f:
                    f.write(response.model_dump_json())
            elif hasattr(response, "to_dict"):
                with open(raw_path, "w") as f:
                    json.dump(response.to_dict(), f)
            else:
                with open(raw_path, "w") as f:
                    json.dump(response, f)
        except Exception as e:
            print(f"Warning: Could not save raw response for {task} seed {seed}: {e}")

    @staticmethod
    def extract_from_response(response: any) -> Optional[str]:
        """
        Extract content from an API response object.

        Args:
            response: API response object

        Returns:
            Response content as string, or None if not available
        """
        if response is None:
            return None

        if not hasattr(response, "choices") or len(response.choices) == 0:
            return None

        if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "content"):
            return response.choices[0].message.content

        return None


class ExtractionStats:
    """Track code extraction statistics."""

    def __init__(self):
        self.success = 0
        self.failed = 0
        self.empty = 0

    def record(self, status: str):
        """Record an extraction result."""
        if status == 'success':
            self.success += 1
        elif status == 'empty':
            self.empty += 1
        else:
            self.failed += 1

    def print_summary(self):
        """Print extraction statistics summary."""
        print("\nCode Extraction Statistics:")
        print(f"  Successful extractions: {self.success}")
        print(f"  Failed extractions: {self.failed}")
        print(f"  Empty code blocks: {self.empty}")

    def __str__(self):
        return f"Success: {self.success}, Failed: {self.failed}, Empty: {self.empty}"
