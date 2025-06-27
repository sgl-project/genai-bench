import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from genai_bench.logging import init_logger
from genai_bench.metrics.metrics import AggregatedMetrics, RequestLevelMetrics
from genai_bench.protocol import ExperimentMetadata

logger = init_logger(__name__)

MetricsData = (
    Dict[Literal["aggregated_metrics"], AggregatedMetrics]
    | Dict[Literal["individual_metrics"], List[RequestLevelMetrics]]
)

ExperimentMetrics = Dict[
    str,  # traffic-scenario
    Dict[
        int,  # concurrency-level
        MetricsData,
    ],
]


def load_multiple_experiments(
    folder_name: str, filter_criteria=None
) -> List[Tuple[ExperimentMetadata, ExperimentMetrics]]:
    """
    Loads the JSON files from one experiment folder. The folder should contain
    a list of subfolders, each subfolder corresponding to an experiment.

    Args:
        folder_name (str): Path to the folder containing the experiment data.
        filter_criteria (dict, optional): Dictionary of filtering criteria based
            on metadata keys.

    Returns:
        list: A list of tuples (run_data, experiment_metadata) from each
            subfolder.
    """
    run_data_list = []

    # Loop through subfolders and files in the folder
    for subfolder in os.listdir(folder_name):
        subfolder_path = os.path.join(folder_name, subfolder)
        if os.path.isdir(subfolder_path):
            # Recursively load data from subfolders
            metadata, run_data = load_one_experiment(subfolder_path, filter_criteria)
            if metadata and run_data:
                run_data_list.append((metadata, run_data))

    return run_data_list


def load_one_experiment(
    folder_name: str, filter_criteria: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[ExperimentMetadata], ExperimentMetrics]:
    """
    Processes files in the provided folder (for metadata and run data).

    Args:
        folder_name (str): Path to the folder.
        filter_criteria (dict, optional): Dictionary of filtering criteria.

    Returns:
        ExperimentMetadata: ExperimentMetadata object.
        dict: Dictionary containing the metrics for each scenario and
            concurrency.
    """
    experiment_file = os.path.join(folder_name, "experiment_metadata.json")

    if not os.path.exists(experiment_file):
        return None, {}

    experiment_metadata = load_experiment_metadata(experiment_file, filter_criteria)
    if not experiment_metadata:
        return None, {}

    run_data: ExperimentMetrics = {}

    for file_name in sorted(os.listdir(folder_name)):
        file_path = os.path.join(folder_name, file_name)
        if re.match(
            r"^.+_.+_(?:concurrency|batch_size)_\d+_time_\d+s\.json$", file_name
        ):
            load_run_data(file_path, run_data, filter_criteria)

    if not run_data:
        return experiment_metadata, run_data

    for scenario in experiment_metadata.traffic_scenario:
        if scenario not in run_data:
            logger.warning(
                f"‼️ Scenario {scenario} in metadata but metrics not found! "
                f"Please re-run this scenario if necessary!"
            )
            experiment_metadata.traffic_scenario.remove(scenario)

    expected_concurrency_list = {
        "batch_size": experiment_metadata.batch_size,
        "num_concurrency": experiment_metadata.num_concurrency,
    }.get(experiment_metadata.iteration_type, [])
    expected_concurrency = set(expected_concurrency_list or [])

    # Check if any scenarios are missing concurrency levels
    for scenario_key, scenario_data in run_data.items():
        seen_concurrency: Set[int] = scenario_data.get(
            f"{experiment_metadata.iteration_type}_levels", set()
        )  # type: ignore[call-overload]
        missing_concurrency: List[Any] = sorted(expected_concurrency - seen_concurrency)
        if missing_concurrency:
            logger.warning(
                f"‼️ Scenario '{scenario_key}' is missing "
                f"{experiment_metadata.iteration_type} levels: {missing_concurrency}. "
                f"Please re-run this scenario if necessary!"
            )
        del scenario_data[f"{experiment_metadata.iteration_type}_levels"]  # type: ignore[arg-type]

    return experiment_metadata, run_data


def load_experiment_metadata(
    file_path: str, filter_criteria: Optional[Dict[str, Any]] = None
) -> Optional[ExperimentMetadata]:
    """
    Loads the experiment metadata from the provided JSON file path.

    Args:
        file_path (str): Path to the `experiment_metadata.json` file.
        filter_criteria (dict, optional): Dictionary of filtering criteria.

    Returns:
        ExperimentMetadata: Filtered ExperimentMetadata object.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        experiment_metadata = ExperimentMetadata(**data)

    if filter_criteria and not apply_filter_to_metadata(
        experiment_metadata, filter_criteria
    ):
        logger.info(
            f"No match with filter_criteria found in ExperimentMetadata under "
            f"{file_path}."
        )
        return None  # Metadata does not match the filter

    return experiment_metadata


def apply_filter_to_metadata(
    experiment_metadata: ExperimentMetadata, filter_criteria: Dict[str, Any]
) -> bool:
    """
    Applies filter criteria to the experiment metadata.

    Args:
        experiment_metadata (ExperimentMetadata): The ExperimentMetadata object
            to filter.
        filter_criteria (dict): The dictionary of filter keys and values.

    Returns:
        bool: True if the metadata matches the filter, False otherwise.
    """
    for key, val in filter_criteria.items():
        if key not in experiment_metadata.model_fields:
            logger.info(f"Filter key {key} is not in the metadata.")
            return False  # Key not present

        if key == "traffic_scenario":
            if not isinstance(val, list):
                val = [val]
            filtered_scenarios = set(experiment_metadata.traffic_scenario).intersection(
                set(val)
            )
            experiment_metadata.traffic_scenario = list(filtered_scenarios)
            if not filtered_scenarios:
                logger.info(
                    f"The scenarios {val} you want to filter is not "
                    f"presented in your experiments."
                )
                return False  # No matching scenarios
        elif getattr(experiment_metadata, key) != val:
            logger.info(
                f"Filter {key}:{val} does not match the value in "
                f"experiment metadata: {getattr(experiment_metadata, key)}"
            )
            return False  # Metadata value doesn't match

    return True


def load_run_data(
    file_path: str,
    run_data: dict,
    filter_criteria: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Loads run data from individual scenario JSON files and filters based on
    criteria.

    Args:
        file_path (str): Path to the JSON file containing metrics.
        run_data (dict): Dictionary where the data will be stored.
        filter_criteria (dict, optional): Filtering criteria.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        aggregated_metrics = AggregatedMetrics(**data["aggregated_metrics"])
        scenario = aggregated_metrics.scenario

        if (
            filter_criteria
            and "traffic_scenario" in filter_criteria
            and scenario not in filter_criteria["traffic_scenario"]
        ):
            return  # Skip if scenario not in the filtered list

        # Get the iteration type and value
        iteration_type = aggregated_metrics.iteration_type
        iteration_value = (
            aggregated_metrics.batch_size
            if iteration_type == "batch_size"
            else aggregated_metrics.num_concurrency
        )

        # Store iteration values in scenario data
        iteration_key = f"{iteration_type}_levels"
        run_data.setdefault(scenario, {}).setdefault(iteration_key, set()).add(
            iteration_value
        )

        # Store the metrics data
        run_data.setdefault(scenario, {})[iteration_value] = {
            "aggregated_metrics": aggregated_metrics,
            "individual_request_metrics": data.get("individual_request_metrics", []),
        }
