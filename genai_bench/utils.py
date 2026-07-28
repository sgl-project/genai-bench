import os
import re
from typing import Union

from genai_bench.logging import init_logger

logger = init_logger(__name__)

_SSL_VERIFY_ENV_VARS = (
    "GENAI_BENCH_SSL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
)


def get_requests_verify() -> Union[bool, str]:
    """Return the SSL verify setting for requests under gevent.

    Gevent monkey-patches ssl before requests runs, so relying on implicit
    CA discovery is unreliable for private CAs. Pass this value as ``verify=``
    on every ``requests`` call.

    Reads, in order: GENAI_BENCH_SSL_CA_BUNDLE, REQUESTS_CA_BUNDLE,
    SSL_CERT_FILE. Falls back to True (default certifi bundle) when unset.
    """
    for env_var in _SSL_VERIFY_ENV_VARS:
        ca_bundle = os.environ.get(env_var)
        if ca_bundle:
            return ca_bundle
    return True


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
