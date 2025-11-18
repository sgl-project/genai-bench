import asyncio
import logging
import sys

import click

from plugin.benchflow.core.config_generator import generate_configs
from plugin.benchflow.core.workflow import run_workflows
from plugin.benchflow.logging import get_logger, setup_logging

logger = get_logger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool):
    """BenchFlow CLI tool for running benchmarks and generating configs."""
    setup_logging(log_level=logging.DEBUG if debug else logging.INFO)
    # Store debug flag in the context for use in subcommands
    ctx = click.get_current_context()
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--param-grid",
    type=click.Path(exists=True),
    help="Path to parameter grid JSON for config generation",
)
@click.option(
    "--output-file",
    type=click.Path(),
    default="plugin/benchflow/generated_configs/combined_config.json",
    help="Path to save the generated config file",
)
def generate(config_path: str, param_grid: str, output_file: str):
    """Generate benchmark configurations using a parameter grid."""
    try:
        output_path = generate_configs(config_path, param_grid, output_file)
        logger.info(f"Successfully generated config at: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate configs: {e}")
        sys.exit(1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path: str):
    """Run inference server and genai-bench workflows defined in CONFIG_PATH."""
    ctx = click.get_current_context()
    debug = ctx.obj["debug"]

    try:
        asyncio.run(run_workflows(config_path, debug))
    except Exception as e:
        logger.error(f"Failed to run workflows: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
