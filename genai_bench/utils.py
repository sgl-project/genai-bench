import os
from pathlib import Path

from transformers import PreTrainedTokenizer


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
    sonnet_file = Path(__file__).parent.resolve() / "sampling/sonnet.txt"
    with open(sonnet_file, "r") as f:
        content = f.read()

    total_chars = len(content)
    tokens = tokenizer.encode(content, add_special_tokens=False)
    total_tokens = len(tokens)

    char_token_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    return char_token_ratio
