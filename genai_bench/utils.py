import os
import re
from pathlib import Path

from transformers import PreTrainedTokenizer

from genai_bench.logging import init_logger

logger = init_logger(__name__)


def sanitize_string(input_str: str):
    """
    Sanitize a string to be used in filenames by replacing problematic
    characters.
    """
    return (
        input_str.replace("/", "_").replace(",", "_").replace("(", "").replace(")", "")
    )


def is_single_experiment_folder(folder_name: str) -> bool:
    """
    Checks whether the folder contains only one experiment by inspecting
    whether the folder has files or subfolders.

    Returns True if it appears to be a single experiment folder, False
    otherwise.
    """
    # Check if the folder has any subdirectories (which would indicate
    # multiple experiments)
    subfolders = [
        f
        for f in os.listdir(folder_name)
        if os.path.isdir(os.path.join(folder_name, f))
    ]
    return len(subfolders) == 0


def calculate_sonnet_char_token_ratio(tokenizer: PreTrainedTokenizer) -> float:
    """Calculate the ratio of character to token using model tokenizer."""
    sonnet_file = Path(__file__).parent.resolve() / "data/sonnet.txt"
    with open(sonnet_file, "r") as f:
        content = f.read()

    total_chars = len(content)
    tokens = tokenizer.encode(content, add_special_tokens=False)
    total_tokens = len(tokens)

    char_token_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    return char_token_ratio


def safe_eval_prompt(prompt_template: str, item: dict) -> str:
    """
    Safely evaluate a prompt template over a dataset row.

    Supports lambda expressions like: lambda x: x["conversations"][0]["content"]
    and simple field access (prompt_template is a key in the item dict).
    """
    # Handle lambda expressions
    template = prompt_template.strip()
    if template.startswith("lambda"):
        try:
            lambda_part, expr = template.split(":", 1)
            var_name = lambda_part.replace("lambda", "").strip()
            expr = expr.strip()
            # Replace x with context
            expr = re.sub(rf"\b{re.escape(var_name)}\b", "context", expr)
            # Safe evaluation by restricting allowed functions
            safe_dict = {"context": item, "str": str, "len": len}
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return str(result) if result is not None else ""
        except Exception as e:
            logger.warning(
                f"Failed to evaluate prompt template: {prompt_template}, error: {e}"
            )
            return ""

    # Simple field access
    if template in item:
        return str(item[template])

    return prompt_template
