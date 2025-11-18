import json
from pathlib import Path

from plugin.benchflow.core.config_generator import ConfigGenerator


def example_direct_api():
    """Example of using ConfigGenerator API directly"""
    # Create generator from base config
    config_path = Path(
        "plugin/benchflow/configs/h100/llama3_2_11b_instruct_text_to_text.json"
    )
    with open(config_path) as f:
        base_config = json.load(f)
    generator = ConfigGenerator(base_config)

    # Define parameter grid for sweeping
    param_grid = {
        "workflows.service.version": ["v0.6.3.post1", "v0.5.3.post1"],
        "workflows.service.extra_args.gpu-memory-utilization": [0.8, 0.9, 0.95],
        "workflows.service.extra_args.tensor-parallel-size": [1, 2, 4],
    }

    # Define parameter links
    linked_params = {
        "workflows.service.version": "workflows.bench.extra_args.server-version"
    }

    # Generate variants
    variants = generator.generate_variants(param_grid, linked_params)

    # Save combined config
    output_path = Path("plugin/benchflow/generated_configs/my_experiment.json")
    generator.save_combined_config(variants, output_path)

    print(f"Generated combined config with {len(variants)} workflows: {output_path}")
    print("Parameter combinations:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")


def example_cli_usage():
    """Example of using benchflow CLI with config generation"""
    print("\nExample CLI commands:")
    print(
        "# Generate configs only:\n"
        "python -m plugin.benchflow.core.benchflow generate "
        "plugin/benchflow/configs/h100/llama3_2_11b_instruct_text_to_text.json "
        "--param-grid plugin/benchflow/examples/param_grid.json "
        "--output-file plugin/benchflow/generated_configs/my_experiment.json"
    )
    print(
        "\n# Run a specific generated config:\n"
        "python -m plugin.benchflow.core.benchflow run "
        "plugin/benchflow/generated_configs/my_experiments/llama3_2_11b_instruct_image_to_text_generated.json"
    )


if __name__ == "__main__":
    print("Example 1: Direct API Usage")
    print("-" * 50)
    example_direct_api()

    print("\nExample 2: CLI Usage")
    print("-" * 50)
    example_cli_usage()
