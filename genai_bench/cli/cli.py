from locust.env import Environment
from locust.runners import WorkerRunner

import os
import sys
import time
from pathlib import Path

import click
import gevent

from genai_bench.analysis.excel_report import create_workbook
from genai_bench.analysis.experiment_loader import load_one_experiment
from genai_bench.analysis.flexible_plot_report import plot_experiment_data_flexible
from genai_bench.analysis.plot_report import (
    plot_single_scenario_inference_speed_vs_throughput,
)
from genai_bench.auth.unified_factory import UnifiedAuthFactory
from genai_bench.cli.option_groups import (
    api_options,
    distributed_locust_options,
    experiment_options,
    metrics_options,
    model_auth_options,
    object_storage_options,
    oci_auth_options,
    sampling_options,
    server_options,
    storage_auth_options,
)
from genai_bench.cli.report import excel, plot
from genai_bench.cli.utils import get_experiment_path, get_run_params, manage_run_time
from genai_bench.cli.validation import validate_tokenizer
from genai_bench.data.config import DatasetConfig
from genai_bench.data.loaders.factory import DataLoaderFactory
from genai_bench.distributed.runner import DistributedConfig, DistributedRunner
from genai_bench.logging import LoggingManager, init_logger
from genai_bench.protocol import ExperimentMetadata
from genai_bench.sampling.base import Sampler
from genai_bench.storage.factory import StorageFactory
from genai_bench.ui.dashboard import create_dashboard
from genai_bench.utils import sanitize_string
from genai_bench.version import __version__ as GENAI_BENCH_VERSION


@click.group()
@click.version_option(
    version=GENAI_BENCH_VERSION,
    prog_name="genai-bench",
    message="%(prog)s version %(version)s",
    help="Show the current version of genai-bench and exit.",
)
@click.pass_context
def cli(ctx):
    """
    Main CLI entry point for genai-bench.
    """
    pass


@click.command(context_settings={"show_default": True})
@api_options
@model_auth_options
@oci_auth_options
@server_options
@experiment_options
@sampling_options
@distributed_locust_options
@object_storage_options
@storage_auth_options
@metrics_options
@click.pass_context
def benchmark(
    ctx,
    api_backend,
    api_base,
    api_key,
    api_model_name,
    model,
    model_tokenizer,
    task,
    iteration_type,
    num_concurrency,
    warmup_ratio,
    cooldown_ratio,
    batch_size,
    traffic_scenario,
    additional_request_params,
    # Model auth options
    model_auth_type,
    model_api_key,
    aws_access_key_id,
    aws_secret_access_key,
    aws_session_token,
    aws_profile,
    aws_region,
    azure_endpoint,
    azure_deployment,
    azure_api_version,
    azure_ad_token,
    gcp_project_id,
    gcp_location,
    gcp_credentials_path,
    # OCI auth options
    config_file,
    profile,
    auth,
    security_token,
    region,
    # Server options
    server_engine,
    server_version,
    server_gpu_type,
    server_gpu_count,
    max_time_per_run,
    max_requests_per_run,
    experiment_folder_name,
    experiment_base_dir,
    log_dir,
    dataset_path,
    dataset_config,
    dataset_prompt_column,
    dataset_image_column,
    prefix_len,
    num_workers,
    master_port,
    spawn_rate,
    upload_results,
    namespace,
    # Storage auth options
    storage_provider,
    storage_bucket,
    storage_prefix,
    storage_auth_type,
    storage_aws_access_key_id,
    storage_aws_secret_access_key,
    storage_aws_session_token,
    storage_aws_region,
    storage_aws_profile,
    storage_azure_account_name,
    storage_azure_account_key,
    storage_azure_connection_string,
    storage_azure_sas_token,
    storage_gcp_project_id,
    storage_gcp_credentials_path,
    github_token,
    github_owner,
    github_repo,
    metrics_time_unit,
    # Metrics options
    metrics_refresh_interval,
):
    """
    Run a benchmark based on user defined scenarios.
    """
    # Set up the dashboard and layout
    dashboard = create_dashboard(metrics_time_unit)

    # Initialize logging with the layout for the log panel
    log_dir = ctx.params.get("log_dir")
    logging_manager = LoggingManager(
        "benchmark", dashboard.layout, dashboard.live, log_dir=log_dir
    )
    delayed_log_handler = logging_manager.delayed_handler
    logger = init_logger("genai_bench.benchmark")

    logger.info(
        f"👋 Welcome to genai-bench {GENAI_BENCH_VERSION}! I am an intelligent "
        "benchmark tool for Large Language Model."
    )

    # Log all parameters
    logger.info("Options you provided:")
    for key, value in ctx.params.items():
        logger.info(f"{key}: {value}")

    # -------------------------------------
    # Authentication Section
    # -------------------------------------

    # Create model authentication based on API backend
    auth_kwargs = {}

    if api_backend == "openai":
        # OpenAI uses API key from --api-key for backward compatibility
        # or --model-api-key for consistency with multi-cloud
        auth_kwargs["api_key"] = model_api_key or api_key

    elif api_backend in ["oci-cohere", "cohere", "oci-genai"]:
        # OCI uses its own auth system
        auth_kwargs.update(
            {
                "auth_type": auth,
                "config_path": config_file,
                "profile": profile,
                "token": security_token,
                "region": region,
            }
        )

    elif api_backend == "aws-bedrock":
        auth_kwargs.update(
            {
                "access_key_id": aws_access_key_id,
                "secret_access_key": aws_secret_access_key,
                "session_token": aws_session_token,
                "profile": aws_profile,
                "region": aws_region,
            }
        )

    elif api_backend == "azure-openai":
        auth_kwargs.update(
            {
                "api_key": model_api_key,
                "api_version": azure_api_version,
                "azure_endpoint": azure_endpoint,
                "azure_deployment": azure_deployment,
                "use_azure_ad": bool(azure_ad_token),
                "azure_ad_token": azure_ad_token,
            }
        )

    elif api_backend == "gcp-vertex":
        auth_kwargs.update(
            {
                "project_id": gcp_project_id,
                "location": gcp_location,
                "credentials_path": gcp_credentials_path,
                "api_key": model_api_key,
            }
        )

    elif api_backend == "together":
        # Together uses API key for authentication
        auth_kwargs["api_key"] = model_api_key or api_key

    elif api_backend in ["vllm", "sglang"]:
        # vLLM and SGLang use OpenAI-compatible API
        auth_kwargs["api_key"] = model_api_key or api_key

    # Map backend names for auth factory
    auth_backend_map = {
        "oci-cohere": "oci",
        "cohere": "oci",
        "oci-genai": "oci",
        "vllm": "openai",
        "sglang": "openai",
    }
    auth_backend = auth_backend_map.get(api_backend, api_backend)

    # Create authentication provider
    auth_provider = UnifiedAuthFactory.create_model_auth(auth_backend, **auth_kwargs)
    logger.info(f"Using {api_backend} authentication")

    # Rebuild the cmd_line from ctx.params
    cmd_line_parts = [sys.argv[0]]
    for key, value in ctx.params.items():
        if isinstance(value, (list, tuple)):
            for item in value:
                cmd_line_parts.append(f"--{key}".replace("_", "-"))
                cmd_line_parts.append(str(item))
        elif value:
            cmd_line_parts.append(f"--{key}".replace("_", "-"))
            cmd_line_parts.append(str(value))
    cmd_line = " ".join(cmd_line_parts)

    user_class = ctx.obj.get("user_class")
    user_task = ctx.obj.get("user_task")

    # Set authentication and API configuration for the user class
    user_class.auth_provider = auth_provider
    user_class.host = api_base
    user_class.api_backend = api_backend

    # Load the tokenizer
    tokenizer = validate_tokenizer(model_tokenizer)

    # Validate prefix_len with full context now that all CLI parameters are available
    from genai_bench.cli.validation import validate_prefix_len_with_context
    validate_prefix_len_with_context(
        prefix_len=prefix_len,
        task=task,
        dataset_path=dataset_path,
        dataset_config=dataset_config,
        traffic_scenario=traffic_scenario,
    )

    # Handle dataset configuration
    if dataset_config:
        # Load from config file
        dataset_config_obj = DatasetConfig.from_file(dataset_config)
    else:
        # Build configuration from CLI arguments
        dataset_config_obj = DatasetConfig.from_cli_args(
            dataset_path=dataset_path,
            prompt_column=dataset_prompt_column,
            image_column=dataset_image_column,
        )

    # Load data using the factory
    data = DataLoaderFactory.load_data_for_task(task, dataset_config_obj)

    # Create sampler with preloaded data
    sampler = Sampler.create(
        task=task,
        tokenizer=tokenizer,
        model=api_model_name,
        data=data,
        additional_request_params=additional_request_params,
        dataset_config=dataset_config_obj,
        prefix_len=prefix_len,
    )

    # If user did not provide scenarios but provided a dataset, default to dataset mode
    if not traffic_scenario and (dataset_path or dataset_config):
        logger.info(
            "No traffic scenarios provided. Using dataset mode: sample raw entries "
            "from the dataset."
        )
        traffic_scenario = ["dataset"]

    max_time_per_run *= 60

    experiment_folder_path: Path = get_experiment_path(
        experiment_folder_name=experiment_folder_name,
        experiment_base_dir=experiment_base_dir,
        api_backend=api_backend,
        server_engine=server_engine,
        server_version=server_version,
        task=task,
        model=model,
    )
    experiment_folder_abs_path = str(experiment_folder_path.absolute())

    logger.info(
        f"This experiment will be saved in folder {experiment_folder_abs_path}."
    )

    experiment_metadata = ExperimentMetadata(
        cmd=cmd_line,
        benchmark_version=GENAI_BENCH_VERSION,
        api_backend=api_backend,
        auth_config=auth_provider.get_config(),
        api_model_name=api_model_name,
        server_model_tokenizer=model_tokenizer,
        model=model,
        task=task,
        num_concurrency=num_concurrency,
        batch_size=batch_size,
        iteration_type=iteration_type,
        traffic_scenario=traffic_scenario,
        server_engine=server_engine,
        server_version=server_version,
        server_gpu_type=server_gpu_type,
        server_gpu_count=server_gpu_count,
        max_time_per_run_s=max_time_per_run,
        max_requests_per_run=max_requests_per_run,
        experiment_folder_name=experiment_folder_abs_path,
        additional_request_params=additional_request_params,
        dataset_path=str(dataset_path),
        metrics_time_unit=metrics_time_unit,
    )
    experiment_metadata_file = Path(
        os.path.join(experiment_folder_abs_path, "experiment_metadata.json")
    )
    experiment_metadata_file.write_text(experiment_metadata.model_dump_json(indent=4))

    # Initialize environment
    environment = Environment(user_classes=[user_class])
    # Assign the selected task to the user class
    environment.user_classes[0].tasks = [user_task]
    environment.sampler = sampler

    # Set up distributed runner
    config = DistributedConfig(
        num_workers=num_workers,
        master_port=master_port,
        log_dir=log_dir,
    )
    runner = DistributedRunner(
        environment=environment,
        config=config,
        dashboard=dashboard,
    )
    runner.setup()

    # Worker process doesn't need to run the main benchmark flow as it only
    # sends requests and collects response
    if num_workers > 0 and isinstance(environment.runner, WorkerRunner):
        return

    # Get metrics collector from runner for master/local mode
    if not runner.metrics_collector:
        raise RuntimeError("Metrics collector not initialized")
    aggregated_metrics_collector = runner.metrics_collector

    # Iterate over each scenario_str and concurrency level,
    # and run the experiment
    iteration_values = batch_size if iteration_type == "batch_size" else num_concurrency
    total_runs = len(traffic_scenario) * len(iteration_values)
    with dashboard.live:
        for scenario_str in traffic_scenario:
            dashboard.reset_plot_metrics()
            sanitized_scenario_str = sanitize_string(scenario_str)
            runner.update_scenario(scenario_str)

            # Store metrics for current scenario for interim plot
            scenario_metrics = {
                "data": {},
                f"{iteration_type}": [],
            }

            for iteration in iteration_values:
                dashboard.reset_panels()
                # Create a new progress bar on dashboard
                iteration_header, batch_size, concurrency = get_run_params(
                    iteration_type, iteration
                )
                dashboard.create_benchmark_progress_task(
                    f"Scenario: {scenario_str}, {iteration_header}: {iteration}"
                )

                # Update batch size for each iteration
                runner.update_batch_size(batch_size)

                aggregated_metrics_collector.set_run_metadata(
                    iteration,
                    scenario_str,
                    metrics_refresh_interval,
                    iteration_type,
                )

                # Start the run
                start_time = time.monotonic()
                dashboard.start_run(max_time_per_run, start_time, max_requests_per_run)

                # Use custom spawn rate if provided, otherwise use concurrency
                actual_spawn_rate = (
                    spawn_rate if spawn_rate is not None else concurrency
                )
                logger.info(
                    f"Starting benchmark with concurrency={concurrency}, "
                    f"spawn_rate={actual_spawn_rate}"
                )
                environment.runner.start(concurrency, spawn_rate=actual_spawn_rate)

                total_run_time = manage_run_time(
                    max_time_per_run=max_time_per_run,
                    max_requests_per_run=max_requests_per_run,
                    environment=environment,
                )

                environment.runner.stop()

                # Aggregate metrics after each run
                end_time = time.monotonic()
                try:
                    aggregated_metrics_collector.aggregate_metrics_data(
                        start_time,
                        end_time,
                        warmup_ratio,
                        cooldown_ratio,
                    )
                except ValueError as e:
                    debug_file_name = (
                        f"debug_for_run_{sanitized_scenario_str}_{concurrency}.json"
                    )
                    aggregated_metrics_collector.save(
                        os.path.join(experiment_folder_abs_path, debug_file_name),
                        metrics_time_unit,
                    )
                    raise ValueError(
                        f"{str(e)} Please check out "
                        f"{debug_file_name} to see the detailed individual "
                        f"metrics!"
                    ) from e

                dashboard.update_scatter_plot_panel(
                    aggregated_metrics_collector.get_ui_scatter_plot_metrics(
                        metrics_time_unit
                    ),
                    metrics_time_unit,
                )

                logger.info(
                    f"⏩ Run for scenario {scenario_str}, "
                    f"{iteration_type} {iteration} has finished after "
                    f"{int(end_time - start_time)} seconds."
                )

                # Save and clear metrics after each run
                run_name = (
                    f"{sanitized_scenario_str}_{task}_{iteration_type}_"
                    f"{iteration}_time_{total_run_time}s.json"
                )
                aggregated_metrics_collector.save(
                    os.path.join(experiment_folder_abs_path, run_name),
                    metrics_time_unit,
                )
                # Store metrics in memory for interim plot
                scenario_metrics["data"][iteration] = {
                    "aggregated_metrics": aggregated_metrics_collector.aggregated_metrics  # noqa: E501
                }
                scenario_metrics[f"{iteration_type}"].append(iteration)

                aggregated_metrics_collector.clear()

                dashboard.update_total_progress_bars(total_runs)

                # Sleep for 1 sec for server to clear aborted requests
                gevent.sleep(1)

            # Plot using in-memory data after all concurrency levels are done
            plot_single_scenario_inference_speed_vs_throughput(
                scenario_str,
                experiment_folder_abs_path,
                task,
                scenario_metrics,
                iteration_type,
            )

        # Sleep for 2 secs before the UI disappears
        gevent.sleep(2)

    # Final cleanup
    runner.cleanup()

    # Flash all the logs to terminal
    if delayed_log_handler:
        delayed_log_handler.flush_buffer()
    logger.info("🚀 The whole experiment has finished!")

    # Generate excel and plots for report
    experiment_metadata, run_data = load_one_experiment(experiment_folder_abs_path)
    create_workbook(
        experiment_metadata,
        run_data,
        os.path.join(
            experiment_folder_abs_path,
            f"{Path(experiment_folder_abs_path).name}_summary.xlsx",
        ),
        percentile="mean",
        metrics_time_unit=metrics_time_unit,
    )
    plot_experiment_data_flexible(
        [
            (experiment_metadata, run_data),
        ],
        group_key="traffic_scenario",
        experiment_folder=experiment_folder_abs_path,
        metrics_time_unit=metrics_time_unit,
    )
    logger.info(
        f"📁 Please check {experiment_folder_abs_path} "
        f"for the detailed results, sheets, and plots."
    )

    # Upload experiment results to object storage
    if not upload_results:
        return

    # Determine storage provider and bucket
    # For backward compatibility, use OCI if storage_provider not specified
    storage_provider_final = storage_provider or "oci"
    storage_bucket_final = storage_bucket
    storage_prefix_final = storage_prefix

    # Create storage authentication
    storage_auth_kwargs = {}

    if storage_provider_final == "oci":
        storage_auth_kwargs.update(
            {
                "auth_type": storage_auth_type or auth,
                "config_path": config_file,
                "profile": profile,
                "token": security_token,
                "region": region,
            }
        )

    elif storage_provider_final == "aws":
        storage_auth_kwargs.update(
            {
                "access_key_id": storage_aws_access_key_id,
                "secret_access_key": storage_aws_secret_access_key,
                "session_token": storage_aws_session_token,
                "region": storage_aws_region,
                "profile": storage_aws_profile,
            }
        )

    elif storage_provider_final == "azure":
        storage_auth_kwargs.update(
            {
                "account_name": storage_azure_account_name,
                "account_key": storage_azure_account_key,
                "connection_string": storage_azure_connection_string,
                "sas_token": storage_azure_sas_token,
            }
        )

    elif storage_provider_final == "gcp":
        storage_auth_kwargs.update(
            {
                "project_id": storage_gcp_project_id,
                "credentials_path": storage_gcp_credentials_path,
            }
        )

    elif storage_provider_final == "github":
        storage_auth_kwargs.update(
            {
                "token": github_token,
                "owner": github_owner,
                "repo": github_repo,
            }
        )

    # Create storage auth provider
    storage_auth_provider = UnifiedAuthFactory.create_storage_auth(
        storage_provider_final, **storage_auth_kwargs
    )

    # Create storage instance
    storage_kwargs = {}
    if storage_provider_final == "oci" and namespace:
        storage_kwargs["namespace"] = namespace

    storage = StorageFactory.create_storage(
        storage_provider_final, storage_auth_provider, **storage_kwargs
    )

    logger.info(
        f"Uploading experiment results from {experiment_folder_abs_path} "
        f"to {storage_provider_final} bucket: {storage_bucket_final}"
    )

    # Upload folder
    storage.upload_folder(
        experiment_folder_abs_path, storage_bucket_final, prefix=storage_prefix_final
    )

    logger.info(
        f"Successfully uploaded experiment results to {storage_provider_final} "
        f"bucket: {storage_bucket_final}"
    )


cli.add_command(benchmark)
cli.add_command(excel)
cli.add_command(plot)

if __name__ == "__main__":
    cli()
