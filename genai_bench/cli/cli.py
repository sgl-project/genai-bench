from locust.env import Environment
from locust.runners import WorkerRunner

import os
import sys
import time
from pathlib import Path

import click

from genai_bench.analysis.excel_report import create_workbook
from genai_bench.analysis.experiment_loader import load_one_experiment
from genai_bench.analysis.plot_report import (
    plot_experiment_data,
    plot_single_scenario_inference_speed_vs_throughput,
)
from genai_bench.auth.factory import AuthFactory
from genai_bench.cli.option_groups import (
    api_options,
    distributed_locust_options,
    experiment_options,
    object_storage_options,
    oci_auth_options,
    sampling_options,
    server_options,
)
from genai_bench.cli.report import excel, plot
from genai_bench.cli.utils import get_experiment_path, get_run_params, manage_run_time
from genai_bench.cli.validation import validate_tokenizer
from genai_bench.distributed.runner import DistributedConfig, DistributedRunner
from genai_bench.logging import LoggingManager, init_logger
from genai_bench.oci_object_storage.os_datastore import OSDataStore
from genai_bench.protocol import ExperimentMetadata
from genai_bench.sampling.base_sampler import Sampler
from genai_bench.sampling.dataset_loader import DatasetConfig, DatasetPath
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
@oci_auth_options
@server_options
@experiment_options
@sampling_options
@distributed_locust_options
@object_storage_options
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
    batch_size,
    traffic_scenario,
    additional_request_params,
    server_engine,
    server_version,
    server_gpu_type,
    server_gpu_count,
    max_time_per_run,
    max_requests_per_run,
    experiment_folder_name,
    experiment_base_dir,
    dataset_path,
    hf_prompt_column_name,
    hf_image_column_name,
    hf_subset,
    hf_split,
    hf_revision,
    dataset_prompt_column_index,
    config_file,
    profile,
    auth,
    security_token,
    region,
    num_workers,
    master_port,
    upload_results,
    bucket,
    namespace,
    prefix,
):
    """
    Run a benchmark based on user defined scenarios.
    """
    # Set up the dashboard and layout
    dashboard = create_dashboard()

    # Initialize logging with the layout for the log panel
    logging_manager = LoggingManager("benchmark", dashboard.layout, dashboard.live)
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

    auth_provider = None
    if api_backend != "openai":
        # Initialize OCI auth provider
        auth_provider = AuthFactory.create_oci_auth(
            auth_type=auth,  # Auth types already match internal types
            config_path=config_file,
            profile=profile,
            token=security_token,
            region=region,
        )
        logger.info(f"Using OCI authentication: {auth}")
    else:
        # Initialize OpenAI auth provider
        auth_provider = AuthFactory.create_openai_auth(api_key)
        logger.info("Using OpenAI authentication")

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

    # Load the tokenizer
    tokenizer = validate_tokenizer(model_tokenizer)

    sonnet_character_token_ratio = calculate_sonnet_char_token_ratio(tokenizer)
    logger.info(f"The average character token ratio is: {sonnet_character_token_ratio}")

    dataset_path = DatasetPath.from_value(dataset_path)
    dataset_config = DatasetConfig(
        dataset_path=dataset_path,
        hf_prompt_column_name=hf_prompt_column_name,
        hf_image_column_name=hf_image_column_name,
        hf_subset=hf_subset,
        hf_split=hf_split,
        hf_revision=hf_revision,
        max_requests_per_run=max_requests_per_run,
        dataset_prompt_column_index=dataset_prompt_column_index,
    )
    sampler = Sampler.create(
        task=task,
        tokenizer=tokenizer,
        model=api_model_name,
        additional_request_params=additional_request_params,
        dataset_config=dataset_config,
    )

    if not sampler.use_scenario:
        logger.info(
            f"No traffic scenario needed for dataset type {dataset_path.type}"
            f" and task {task}. Setting scenario to a no-op placeholder 'F' "
            f"(No Effect)."
        )
        traffic_scenario = ["F"]

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
        dataset_prompt_column_index=dataset_prompt_column_index,
        character_token_ratio=sonnet_character_token_ratio,
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
                    iteration, scenario_str, iteration_type
                )

                # Start the run
                start_time = time.monotonic()
                dashboard.start_run(max_time_per_run, start_time, max_requests_per_run)

                environment.runner.start(concurrency, spawn_rate=concurrency)

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
                    )
                except ValueError as e:
                    debug_file_name = (
                        f"debug_for_run_{sanitized_scenario_str}_{concurrency}.json"
                    )
                    aggregated_metrics_collector.save(
                        os.path.join(experiment_folder_abs_path, debug_file_name)
                    )
                    raise ValueError(
                        f"{str(e)} Please check out "
                        f"{debug_file_name} to see the detailed individual "
                        f"metrics!"
                    ) from e

                dashboard.update_scatter_plot_panel(
                    aggregated_metrics_collector.get_ui_scatter_plot_metrics()
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
                    os.path.join(experiment_folder_abs_path, run_name)
                )
                # Store metrics in memory for interim plot
                scenario_metrics["data"][iteration] = {
                    "aggregated_metrics": aggregated_metrics_collector.aggregated_metrics  # noqa: E501
                }
                scenario_metrics[f"{iteration_type}"].append(iteration)

                aggregated_metrics_collector.clear()

                dashboard.update_total_progress_bars(total_runs)

                # Sleep for 5 secs for server to clear aborted requests
                time.sleep(5)

            # Plot using in-memory data after all concurrency levels are done
            plot_single_scenario_inference_speed_vs_throughput(
                scenario_str,
                experiment_folder_abs_path,
                task,
                scenario_metrics,
                iteration_type,
            )

        # Sleep for 5 secs before the UI disappears
        time.sleep(5)

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
    )
    plot_experiment_data(
        [
            (experiment_metadata, run_data),
        ],
        group_key="traffic_scenario",
        experiment_folder=experiment_folder_abs_path,
    )
    logger.info(
        f"📁 Please check {experiment_folder_abs_path} "
        f"for the detailed results, sheets, and plots."
    )

    # Upload experiment results to object storage
    if not upload_results:
        return

    # instead of relying on existing inference auth provider
    # create a new oci auth provider for data store
    data_store_auth_provider = AuthFactory.create_oci_auth(
        auth_type=auth,
        config_path=config_file,
        profile=profile,
        token=security_token,
        region=region,
    )
    casper = OSDataStore(data_store_auth_provider)

    # Override regions/namespace if provided
    casper.set_region(region) if region else None
    namespace = namespace or casper.get_namespace()

    logger.info(
        f"Uploading experiment results from {experiment_folder_abs_path} "
        f"to bucket: {bucket} in namespace {namespace}"
    )
    casper.upload_folder(
        experiment_folder_abs_path, bucket, prefix=prefix, namespace=namespace
    )
    logger.info(f"Successfully uploaded experiment results to bucket: {bucket}")


cli.add_command(benchmark)
cli.add_command(excel)
cli.add_command(plot)

if __name__ == "__main__":
    cli()
