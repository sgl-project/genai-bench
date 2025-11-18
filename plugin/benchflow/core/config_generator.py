import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

from plugin.benchflow.core.protocol import ExecutionMode, WorkflowConfig, WorkflowPair
from plugin.benchflow.logging import get_logger

logger = get_logger(__name__)

LINKED_PARAMS = {
    "workflows.service.version": ["workflows.bench.extra_args.server-version"],
    "workflows.service.extra_args.model": [
        "workflows.bench.extra_args.model-tokenizer"
    ],
    "workflows.service.extra_args.served-model-name": [
        "workflows.bench.extra_args.api-model-name"
    ],
    "workflows.service.extra_args.tensor-parallel-size": [
        "workflows.service.num_gpu_devices",
        "workflows.bench.extra_args.server-gpu-count",
    ],
}


def generate_configs(
    base_config_path: str,
    param_grid_path: str,
    output_file: Optional[
        str
    ] = "plugin/benchflow/generated_configs/combined_config.json",
) -> str:
    """Generate workflow configs using parameter grid.

    Args:
        base_config_path: Path to base config file
        param_grid_path: Path to parameter grid JSON
        output_file: Path to save the generated config file

    Returns:
        Path to generated combined config file
    """
    # Load base config
    with open(base_config_path) as f:
        base_config = json.load(f)

    # Load parameter grid
    with open(param_grid_path) as f:
        param_grid = json.load(f)

    # Create generator and generate variants
    generator = ConfigGenerator(base_config)
    variants = generator.generate_variants(param_grid)

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save combined config
    generator.save_combined_config(variants, output_path)
    logger.info(f"Generated combined config with {len(variants)} variants")

    return str(output_path)


class ConfigGenerator:
    """Generate workflow configurations for parameter sweeps and HPO"""

    def __init__(self, base_config: dict):
        """
        Args:
            base_config: Base configuration with placeholders for parameters to sweep
        """
        self.base_config = base_config

    def generate_variants(
        self,
        param_grid: Dict[str, List[Any]],
        linked_params: Optional[Dict[str, List[str]]] = None,
    ) -> List[dict]:
        """
        Generate config variants based on parameter grid.

        Args:
            param_grid: Dictionary mapping parameter paths to lists of values
            linked_params: Dictionary mapping parameter paths to lists of linked paths
                Example:
                {
                    "workflows.service.version": [
                        "workflows.bench.extra_args.server_version"
                    ],
                    "workflows.service.extra_args.tensor-parallel-size": [
                        "workflows.service.num_gpu_devices",
                        "workflows.bench.extra_args.server-gpu-count"
                    ]
                }

        Returns:
            List of configuration variants
        """
        variants = []
        keys = param_grid.keys()
        values = param_grid.values()
        linked_params = {**LINKED_PARAMS, **(linked_params or {})}

        for i, combination in enumerate(product(*values)):
            config = deepcopy(self.base_config)
            params = dict(zip(keys, combination, strict=False))

            # Update config with new parameters
            for path, value in params.items():
                # Update the main parameter
                self._update_config_value(config, path, value)

                # Update any linked parameters
                if path in linked_params:
                    for linked_path in linked_params[path]:
                        # For server version, add "v" prefix if not present
                        if "server_version" in linked_path and not str(
                            value
                        ).startswith("v"):
                            linked_value = f"v{value}"
                        else:
                            linked_value = value
                        self._update_config_value(config, linked_path, linked_value)

            # Update container names for this variant
            workflow = config["workflows"][0]
            base_name = workflow["name"]

            # Update service and benchmark container name
            workflow["service"]["container_name"] = (
                f"{base_name}_{workflow['service'].get('container_name', 'service')}"
                f"_variant_{i}"
            )
            workflow["bench"]["container_name"] = (
                f"{base_name}_{workflow['bench'].get('container_name', 'bench')}"
                f"_{base_name}_variant_{i}"
            )

            variants.append(config)
        return variants

    def save_combined_config(self, variants: List[dict], output_path: Path) -> None:
        """Save variants as a combined config file"""
        combined_config = WorkflowConfig(
            workflows=[],
            execution_mode=ExecutionMode(
                self.base_config.get("execution_mode", "sequential")
            ),
            max_parallel=self.base_config.get("max_parallel", 2),
        )

        for i, variant in enumerate(variants):
            workflow = variant["workflows"][0]
            workflow["name"] = f"{workflow['name']}_variant_{i}"
            # Convert dict to WorkflowPair object
            workflow_pair = WorkflowPair(**workflow)
            combined_config.workflows.append(workflow_pair)

        # Convert to dict and save as JSON
        config_dict = combined_config.model_dump(mode="json")
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def _update_config_value(self, config: dict, path: str, value: Any) -> None:
        """Update config value using a path string"""
        # Handle workflows array specially
        parts = path.split(".")
        if parts[0] == "workflows":
            # Get the first workflow if no index specified
            workflow = config["workflows"][0]
            remaining_path = parts[1:]  # Skip 'workflows' part

            # Navigate to the correct section (service or bench)
            current = workflow
            for part in remaining_path[:-1]:
                if part == "extra_args":
                    # Handle extra_args as a list
                    current = self._find_or_update_arg(
                        current["extra_args"], remaining_path[-1], value
                    )
                    return
                current = current.setdefault(part, {})

            current[remaining_path[-1]] = value
        else:
            # For non-workflow paths, use normal nested dict update
            self._update_nested_dict(config, parts, value)

    @staticmethod
    def _find_or_update_arg(
        args: List[str], param_name: str, new_value: Any
    ) -> List[str]:
        """Find and update a command line argument in extra_args list"""
        # Remove '--' prefix if present in param_name
        param_name = param_name.lstrip("-")
        prefix = f"--{param_name}"

        # Find and update existing argument
        for i, arg in enumerate(args):
            if arg.startswith(prefix):
                args[i] = f"--{param_name}={new_value}"
                return args

        # Add new argument if not found
        args.append(f"--{param_name}={new_value}")
        return args

    @staticmethod
    def _update_nested_dict(d: dict, keys: List[str], value: Any) -> None:
        """Update nested dictionary using key list"""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
