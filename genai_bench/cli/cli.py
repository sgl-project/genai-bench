from locust.env import Environment
from locust.runners import WorkerRunner

import os
import sys
import time
from pathlib import Path

import click

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
    execution_engine_options,
    experiment_options,
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
from genai_bench.utils import calculate_sonnet_char_token_ratio, sanitize_string
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
@execution_engine_options
@object_storage_options
@storage_auth_options
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
    disable_streaming,
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
    num_workers,
    master_port,
    spawn_rate,
    execution_engine,
    qps_level,
    distribution,
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

    # Log all parameters (filtering out sensitive information)
    logger.info("Options you provided:")
    sensitive_params = {
        "api_key",
        "model_api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "azure_ad_token",
        "github_token",
    }

    for key, value in ctx.params.items():
        # Check if this is a sensitive parameter that came from environment variable
        if key in sensitive_params and os.getenv(key.upper().replace("-", "_")):
            logger.info(f"{key}: [ENV_VAR]")
        else:
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

    elif api_backend == "baseten":
        # Baseten uses API key authentication
        # Check for API key in order: model_api_key, api_key, MODEL_API_KEY, BASETEN_API_KEY
        baseten_api_key = (
            model_api_key
            or api_key
            or os.getenv("MODEL_API_KEY")
            or os.getenv("BASETEN_API_KEY")
        )
        auth_kwargs["api_key"] = baseten_api_key

    # Map backend names for auth factory
    auth_backend_map = {
        "oci-cohere": "oci",
        "cohere": "oci",
        "oci-genai": "oci",
        "vllm": "openai",
        "sglang": "openai",
        "baseten": "baseten",
    }
    auth_backend = auth_backend_map.get(api_backend, api_backend)

    # Create authentication provider
    auth_provider = UnifiedAuthFactory.create_model_auth(auth_backend, **auth_kwargs)
    logger.info(f"Using {api_backend} authentication")

    # Rebuild the cmd_line from ctx.params, filtering out sensitive information
    cmd_line_parts = [sys.argv[0]]
    sensitive_params = {
        "api_key",
        "model_api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "azure_ad_token",
        "github_token",
    }

    for key, value in ctx.params.items():
        if isinstance(value, (list, tuple)):
            for item in value:
                cmd_line_parts.append(f"--{key}".replace("_", "-"))
                # Check if this is a sensitive parameter that came from environment variable
                if key in sensitive_params and os.getenv(key.upper().replace("-", "_")):
                    cmd_line_parts.append("[ENV_VAR]")
                else:
                    cmd_line_parts.append(str(item))
        elif value:
            cmd_line_parts.append(f"--{key}".replace("_", "-"))
            # Check if this is a sensitive parameter that came from environment variable
            if key in sensitive_params and os.getenv(key.upper().replace("-", "_")):
                cmd_line_parts.append("[ENV_VAR]")
            else:
                cmd_line_parts.append(str(value))
    cmd_line = " ".join(cmd_line_parts)

    user_class = ctx.obj.get("user_class")
    user_task = ctx.obj.get("user_task")

    # Set authentication and API configuration for the user class
    user_class.auth_provider = auth_provider
    user_class.host = api_base
    user_class.disable_streaming = disable_streaming
    user_class.api_backend = api_backend

    # Load the tokenizer
    tokenizer = validate_tokenizer(model_tokenizer)

    sonnet_character_token_ratio = calculate_sonnet_char_token_ratio(tokenizer)
    logger.info(f"The average character token ratio is: {sonnet_character_token_ratio}")

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
        character_token_ratio=sonnet_character_token_ratio,
        metrics_time_unit=metrics_time_unit,
    )
    experiment_metadata_file = Path(
        os.path.join(experiment_folder_abs_path, "experiment_metadata.json")
    )
    experiment_metadata_file.write_text(experiment_metadata.model_dump_json(indent=4))

    # Route to appropriate execution engine
    if execution_engine.lower() == "async":
        # Warn if backend not fully supported
        if api_backend not in ["openai", "baseten"]:
            logger.warning(
                f"⚠️ Async runner currently supports OpenAI-compatible backends only. "
                f"Backend '{api_backend}' may not work correctly. "
                f"Consider using --execution-engine=locust instead."
            )

        # Determine async runner execution mode
        # Open-loop: QPS input (--qps-level specified)
        # Closed-loop: Concurrency input (--num-concurrency specified, no --qps-level)
        # Handle qps_level as tuple when multiple values provided, or single value/None
        qps_level_list = (
            list(qps_level)
            if isinstance(qps_level, (tuple, list))
            else ([qps_level] if qps_level is not None else [])
        )

        if not qps_level_list and not num_concurrency:
            raise ValueError(
                "Async runner requires either --qps-level (open-loop mode) or --num-concurrency "
                "(closed-loop mode). Use --qps-level for QPS-based load, or --num-concurrency "
                "for concurrency-based load with better streaming metrics than Locust."
            )

        use_closed_loop = not qps_level_list and num_concurrency is not None

        if use_closed_loop:
            logger.info(
                "🔄 Using async runner in closed-loop mode: maintaining target concurrency "
                "(better streaming metrics than Locust)"
            )
        else:
            logger.info(
                "📊 Using async runner in open-loop mode: fixed QPS arrival rate"
            )

        # Use async runner factory
        from genai_bench.async_runner.factory import create_runner

        # Create aggregated metrics collector for async runner
        from genai_bench.metrics.aggregated_metrics_collector import (
            AggregatedMetricsCollector,
        )

        aggregated_metrics_collector = AggregatedMetricsCollector()

        # Iterate over each scenario and run with async runner
        # For open-loop mode with qps_level, use QPS as iteration value instead of concurrency
        if not use_closed_loop and qps_level_list:
            # Open-loop mode: iterate over QPS values
            iteration_values = qps_level_list
            iteration_type_display = "qps"
        else:
            # Closed-loop mode or batch_size: use normal iteration
            iteration_values = (
                batch_size if iteration_type == "batch_size" else num_concurrency
            )
            iteration_type_display = iteration_type

        # Create runner once - QPS value is passed to run() method, not constructor
        # For open-loop mode, pass first QPS value just to determine runner type (not stored)
        # For closed-loop mode, pass first concurrency value just to determine runner type (not stored)
        if use_closed_loop:
            runner = create_runner(
                sampler=sampler,
                api_backend=api_backend,
                api_base=api_base,
                api_model_name=api_model_name,
                auth_provider=auth_provider,
                aggregated_metrics_collector=aggregated_metrics_collector,
                dashboard=dashboard,
                qps_level=None,
                target_concurrency=num_concurrency[0]
                if num_concurrency
                else None,  # Just for validation, actual value set in run()
            )
        else:
            # Open-loop mode: pass first QPS value just to determine runner type
            # Actual QPS values are passed to run() method per-iteration
            runner = create_runner(
                sampler=sampler,
                api_backend=api_backend,
                api_base=api_base,
                api_model_name=api_model_name,
                auth_provider=auth_provider,
                aggregated_metrics_collector=aggregated_metrics_collector,
                dashboard=dashboard,
                qps_level=qps_level_list[0]
                if qps_level_list
                else None,  # Just for validation, actual value set in run()
                target_concurrency=None,
            )

        total_runs = len(traffic_scenario) * len(iteration_values)
        dashboard.initialize_total_progress_bars(total_runs)

        with dashboard.live:
            for scenario_str in traffic_scenario:
                dashboard.reset_plot_metrics()
                sanitized_scenario_str = sanitize_string(scenario_str)

                # Store metrics for current scenario for interim plot
                scenario_metrics = {
                    "data": {},
                    f"{iteration_type_display}": [],
                }

                # Reset prefix cache for new scenario
                if hasattr(sampler, "reset_prefix_cache"):
                    sampler.reset_prefix_cache()

                for iteration in iteration_values:
                    dashboard.reset_panels()
                    # Create a new progress bar on dashboard
                    if not use_closed_loop and qps_level_list:
                        # Open-loop mode: show QPS
                        iteration_header = "QPS"
                        batch_size = 1
                        concurrency = None
                        progress_label = (
                            f"Scenario: {scenario_str}, {iteration_header}: {iteration}"
                        )
                    else:
                        # Closed-loop mode or batch_size: use normal logic
                        iteration_header, batch_size, concurrency = get_run_params(
                            iteration_type, iteration
                        )
                        progress_label = (
                            f"Scenario: {scenario_str}, {iteration_header}: {iteration}"
                        )

                    dashboard.create_benchmark_progress_task(progress_label)

                    # Clear metrics for new run (must be done before setting metadata)
                    aggregated_metrics_collector.clear()

                    # Set run metadata (after clear to ensure it's not wiped)
                    # For QPS (open-loop mode), store QPS value directly in num_concurrency field
                    # This matches the simpler approach: store decimals as-is, handle in filenames/regex
                    if not use_closed_loop and qps_level_list:
                        # Store QPS value directly (0.5 stays 0.5) in num_concurrency field
                        # Use iteration_type="num_concurrency" for compatibility with existing code
                        aggregated_metrics_collector.set_run_metadata(
                            iteration=iteration,  # Store QPS value as-is (e.g., 0.5)
                            scenario_str=scenario_str,
                            iteration_type="num_concurrency",  # Use existing field for compatibility
                        )
                    else:
                        aggregated_metrics_collector.set_run_metadata(
                            iteration=iteration,
                            scenario_str=scenario_str,
                            iteration_type=iteration_type,
                        )

                    start_time = time.monotonic()

                    # Run with async runner (either open-loop or closed-loop mode)
                    if use_closed_loop:
                        # Closed-loop mode: maintain target concurrency
                        runner.run(
                            target_concurrency=iteration,
                            duration_s=max_time_per_run,
                            distribution=distribution.lower(),
                            random_seed=42,  # Fixed seed for reproducibility
                            max_requests=max_requests_per_run,
                            max_time_s=max_time_per_run,
                            scenario=scenario_str,
                        )
                    else:
                        # Open-loop mode: fixed QPS (use current iteration value)
                        runner.run(
                            qps_level=iteration,  # Current QPS value from iteration_values
                            duration_s=max_time_per_run,
                            distribution=distribution.lower(),
                            random_seed=42,  # Fixed seed for reproducibility
                            max_requests=max_requests_per_run,
                            max_time_s=max_time_per_run,
                            scenario=scenario_str,
                        )

                    end_time = time.monotonic()

                    # Aggregate metrics after each run
                    try:
                        aggregated_metrics_collector.aggregate_metrics_data(
                            start_time,
                            end_time,
                            sonnet_character_token_ratio,
                            warmup_ratio,
                            cooldown_ratio,
                        )
                    except ValueError as e:
                        debug_file_name = (
                            f"debug_for_run_{sanitized_scenario_str}_{iteration}.json"
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
                        f"{iteration_type_display} {iteration} has finished after "
                        f"{int(end_time - start_time)} seconds."
                    )

                    # Save and clear metrics after each run
                    # For QPS, use iteration_type for filename to maintain compatibility
                    # Store QPS values as decimals in filename (handled by regex pattern)
                    filename_iteration_type = (
                        iteration_type
                        if iteration_type_display == "qps"
                        else iteration_type_display
                    )
                    run_name = (
                        f"{sanitized_scenario_str}_{task}_{filename_iteration_type}_"
                        f"{iteration}_time_{max_time_per_run}s.json"
                    )
                    aggregated_metrics_collector.save(
                        os.path.join(experiment_folder_abs_path, run_name),
                        metrics_time_unit,
                    )
                    # Store metrics in memory for interim plot
                    # Use QPS value as-is (decimals) as key to match what's stored in num_concurrency
                    # This ensures consistency between in-memory storage and file loading
                    scenario_metrics["data"][iteration] = {
                        "aggregated_metrics": aggregated_metrics_collector.aggregated_metrics
                    }
                    scenario_metrics[f"{iteration_type_display}"].append(iteration)
                    aggregated_metrics_collector.clear()

                    dashboard.update_total_progress_bars(total_runs)

                    # Sleep for 1 sec for server to clear aborted requests
                    time.sleep(1)

                # Plot using in-memory data after all concurrency levels are done
                plot_single_scenario_inference_speed_vs_throughput(
                    scenario_str,
                    experiment_folder_abs_path,
                    task,
                    scenario_metrics,
                    iteration_type_display,
                )

        # Sleep for 2 secs before the UI disappears
        time.sleep(2)

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
        if upload_results:
            # Determine storage provider and bucket
            storage_provider_final = storage_provider or "oci"
            storage_bucket_final = storage_bucket
            storage_prefix_final = storage_prefix

            # Create storage authentication
            storage_auth_kwargs = {}
            if storage_provider_final == "oci":
                storage_auth_kwargs.update(
                    {
                        "auth_type": storage_auth_type or auth,
                        "config_file": config_file,
                        "profile": profile,
                        "security_token": security_token,
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

            # Create storage instance and upload
            storage_kwargs = {}
            if storage_provider_final == "oci" and namespace:
                storage_kwargs["namespace"] = namespace

            storage = StorageFactory.create_storage(
                storage_provider_final, storage_auth_provider, **storage_kwargs
            )
            storage.upload_directory(
                experiment_folder_abs_path,
                storage_bucket_final,
                storage_prefix_final,
            )
            logger.info(
                f"✅ Results uploaded to {storage_provider_final}://{storage_bucket_final}/{storage_prefix_final}"
            )

        return

    # Initialize environment (Locust path)
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
    dashboard.initialize_total_progress_bars(total_runs)
    with dashboard.live:
        for scenario_str in traffic_scenario:
            dashboard.reset_plot_metrics()
            sanitized_scenario_str = sanitize_string(scenario_str)
            runner.update_scenario(scenario_str)

            # Reset prefix cache for new scenario to ensure fresh prefix
            # This is critical for prefix repetition scenarios to work correctly
            if hasattr(sampler, "reset_prefix_cache"):
                sampler.reset_prefix_cache()

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
                    iteration, scenario_str, iteration_type
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
                        sonnet_character_token_ratio,
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
                time.sleep(1)

            # Plot using in-memory data after all concurrency levels are done
            plot_single_scenario_inference_speed_vs_throughput(
                scenario_str,
                experiment_folder_abs_path,
                task,
                scenario_metrics,
                iteration_type,
            )

        # Sleep for 2 secs before the UI disappears
        time.sleep(2)

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
