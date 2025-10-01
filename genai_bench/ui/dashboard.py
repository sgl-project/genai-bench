import os
import time
from typing import Dict, List, Optional, Union

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from genai_bench.logging import init_logger
from genai_bench.protocol import LiveMetricsData
from genai_bench.ui.layout import (
    create_layout,
    create_metric_panel,
    create_progress_bars,
    update_progress,
)
from genai_bench.ui.plots import (
    create_horizontal_colored_bar_chart,
    create_scatter_plot,
)

logger = init_logger(__name__)


class MinimalDashboard:
    """A minimal implementation of the Dashboard interface for scenarios without UI."""

    def __init__(self, time_unit: str = "s"):
        self.console = None
        self.layout = None
        self.time_unit = time_unit
        self._live = type(
            "MinimalDashboardLive",
            (),
            {
                "__enter__": lambda x: None,
                "__exit__": lambda x, *args: None,
            },
        )()  # a simple no-op context manager to work with dashboard.live in cli.py

    @property
    def live(self):
        return self._live

    def update_metrics_panels(
        self, live_metrics: LiveMetricsData, time_unit: str = "s"
    ):
        pass

    def update_histogram_panel(
        self, live_metrics: LiveMetricsData, time_unit: str = "s"
    ):
        pass

    def update_scatter_plot_panel(
        self, ui_scatter_plot_metrics: Optional[List[float]], time_unit: str = "s"
    ):
        pass

    def update_benchmark_progress_bars(self, progress_increment: float):
        pass

    def create_benchmark_progress_task(self, run_name: str):
        pass

    def update_total_progress_bars(self, total_runs: int):
        pass

    def start_run(self, run_time: int, start_time: float, max_requests_per_run: int):
        pass

    def calculate_time_based_progress(self) -> float:
        return 0.0

    def handle_single_request(
        self, live_metrics: LiveMetricsData, total_requests: int, error_code: int | None
    ):
        pass

    def reset_plot_metrics(self):
        pass

    def reset_panels(self):
        pass


class RichLiveDashboard:
    """
    A dashboard implementation leveraging the `rich` library for live updates and
    interactive UI.

    The RichLiveDashboard provides real-time metrics visualization and progress tracking
    for benchmarking scenarios. It dynamically updates panels, histograms, and scatter
    plots to display key performance data, offering an intuitive and visually appealing
    experience.

    Features:
    - **Rich UI Components**: Uses the `rich` library to create visually rich layouts
      with panels, progress bars, and charts.
    - **Live Updates**: Dynamically updates metrics, histograms, and scatter plots in
      real time.
    - **Interactive Visualization**: Supports the visualization of performance metrics
      like latency, throughput, and task progress.
    - **Custom Layouts**: Provides a structured layout to organize data effectively for
      users.

    Typical Use Case:
    - Designed for benchmarking scenarios where real-time feedback on metrics is
      essential.
    - Enables monitoring of both system performance and task progress.

    Key Panels and Visuals:
    - **Metrics Panels**: Displays input and output latency/throughput metrics.
    - **Histograms**: Visualizes latency distributions with horizontal bar charts.
    - **Scatter Plots**: Correlates throughput and latency for input and output
      phases.
    - **Progress Bars**: Tracks overall benchmarking progress and individual run
      progress.

    Initialization Parameters:
    - No parameters are required. The dashboard initializes with default layouts
      and refresh rates.

    Example:
        dashboard = RichLiveDashboard()
        with dashboard.live:
            dashboard.update_metrics_panels(live_metrics)
            dashboard.update_benchmark_progress_bars(progress_increment)
            ...

    Dependencies:
    - `rich` library for UI and visualization components.
    - `genai_bench` modules for metrics and logging.

    Notes:
    - This class assumes the availability of metrics in the form of `LiveMetricsData`.
    - If `ENABLE_UI` is disabled, consider using `MinimalDashboard` instead.
    """

    def __init__(self, time_unit: str = "s"):
        self.console: Console = Console()
        self.layout = create_layout()
        (
            self.total_progress,
            self.benchmark_progress,
            self.total_progress_task_id,
        ) = create_progress_bars()
        self.benchmark_progress_task_id: Optional[int] = None
        self.start_time: Optional[float] = None
        self.run_time: Optional[int] = None
        self.max_requests_per_run: Optional[int] = None
        self.time_unit: str = time_unit
        self.plot_metrics: Dict[str, List[float]] = {
            "ttft": [],
            "input_throughput": [],
            "output_throughput": [],
            "output_latency": [],
        }
        self.live: Live = Live(
            self.layout,
            console=self.console,
            refresh_per_second=2,
            screen=True,
        )

    def update_metrics_panels(
        self, live_metrics: LiveMetricsData, time_unit: str = "s"
    ):
        if "stats" not in live_metrics or not live_metrics["stats"]:
            return
        stats = live_metrics["stats"]
        # Handle both dict and list formats for stats
        if isinstance(stats, dict):
            input_latency_panel, input_throughput_panel = create_metric_panel(
                "Input",
                stats.get("ttft", []),
                stats.get("input_throughput", []),
                time_unit,
            )
            output_latency_panel, output_throughput_panel = create_metric_panel(
                "Output",
                stats.get("output_latency", []),
                stats.get("output_throughput", []),
                time_unit,
            )
        else:
            # If stats is a list or other format, use empty lists as fallback
            input_latency_panel, input_throughput_panel = create_metric_panel(
                "Input", [], [], time_unit
            )
            output_latency_panel, output_throughput_panel = create_metric_panel(
                "Output", [], [], time_unit
            )

        self.layout["input_throughput"].update(input_throughput_panel)
        self.layout["input_latency"].update(input_latency_panel)
        self.layout["output_throughput"].update(output_throughput_panel)
        self.layout["output_latency"].update(output_latency_panel)

    def update_histogram_panel(
        self, live_metrics: LiveMetricsData, time_unit: str = "s"
    ):
        input_latency_hist_chart = create_horizontal_colored_bar_chart(
            live_metrics["ttft"], bin_width=0.01, time_unit=time_unit
        )
        output_latency_hist_chart = create_horizontal_colored_bar_chart(
            live_metrics["output_latency"],
            bin_width=0.01,
            time_unit=time_unit,
        )

        self.layout["input_histogram"].update(
            Panel(
                input_latency_hist_chart,
                title="Input Latency Histogram",
                border_style="bold green",
                expand=True,
            )
        )
        self.layout["output_histogram"].update(
            Panel(
                output_latency_hist_chart,
                title="Output Latency Histogram",
                border_style="bold blue",
                expand=True,
            )
        )

    def update_scatter_plot_panel(
        self, ui_scatter_plot_metrics: Optional[List[float]], time_unit: str = "s"
    ):
        if not ui_scatter_plot_metrics:
            logger.info("No ui_scatter_plot_metrics collected for this run.")
            return

        ttft, output_latency, input_throughput, output_throughput = (
            ui_scatter_plot_metrics
        )
        self.plot_metrics["ttft"].append(ttft)
        self.plot_metrics["input_throughput"].append(input_throughput)
        self.plot_metrics["output_throughput"].append(output_throughput)
        self.plot_metrics["output_latency"].append(output_latency)

        input_throughput_scatter_plot = create_scatter_plot(
            self.plot_metrics["input_throughput"],
            self.plot_metrics["ttft"],
            y_unit=time_unit,
            x_unit="tokens/sec",
        )
        output_latency_scatter_plot = create_scatter_plot(
            self.plot_metrics["output_throughput"],
            self.plot_metrics["output_latency"],
            y_unit=time_unit,
            x_unit="tokens/sec",
        )

        self.layout["ttft_vs_input_throughput"].update(
            Panel(
                input_throughput_scatter_plot,
                title="Input Latency vs Input Throughput of Server",
                border_style="bold green",
                expand=True,
            )
        )
        self.layout["output_latency_vs_output_throughput"].update(
            Panel(
                output_latency_scatter_plot,
                title="Output Latency vs Output Throughput of Server",
                border_style="bold blue",
                expand=True,
            )
        )

    def update_benchmark_progress_bars(self, progress_increment: float):
        self.benchmark_progress.update(
            self.benchmark_progress_task_id, completed=progress_increment
        )
        update_progress(self.layout, self.total_progress, self.benchmark_progress)

    def create_benchmark_progress_task(self, run_name: str):
        self.benchmark_progress_task_id = self.benchmark_progress.add_task(
            run_name, total=100
        )
        update_progress(self.layout, self.total_progress, self.benchmark_progress)

    def update_total_progress_bars(self, total_runs: int):
        self.benchmark_progress.remove_task(self.benchmark_progress_task_id)
        self.total_progress.update(
            self.total_progress_task_id, advance=(1 / total_runs) * 100
        )
        update_progress(self.layout, self.total_progress, self.benchmark_progress)

    def start_run(self, run_time: int, start_time: float, max_requests_per_run: int):
        self.start_time = start_time
        self.run_time = run_time
        self.max_requests_per_run = max_requests_per_run

    def calculate_time_based_progress(self) -> float:
        assert self.start_time is not None and self.run_time is not None
        time_elapsed = time.monotonic() - self.start_time
        return min(time_elapsed / self.run_time, 1) * 100

    def handle_single_request(
        self,
        live_metrics: LiveMetricsData,
        total_requests: int,
        error_code: int | None,
    ):
        # Calculate time-based progress
        time_based_progress = self.calculate_time_based_progress()

        # Calculate request-based progress
        assert self.max_requests_per_run is not None
        request_based_progress = (
            min(total_requests / self.max_requests_per_run, 1) * 100
        )

        # Use the larger of the two progress metrics to be more accurate
        progress_increment = max(time_based_progress, request_based_progress)

        self.update_benchmark_progress_bars(progress_increment)

        # No need to update metrics panel or histogram panel when the request
        # fails
        if error_code is not None:
            return

        self.update_metrics_panels(live_metrics, self.time_unit)
        self.update_histogram_panel(live_metrics, self.time_unit)

    def reset_plot_metrics(self):
        """Reset plot metrics for each scenario."""
        self.plot_metrics = {
            "ttft": [],
            "input_throughput": [],
            "output_throughput": [],
            "output_latency": [],
        }

        self.layout["ttft_vs_input_throughput"].update(
            Panel(
                Text(),
                title="Input Latency vs Input Throughput of Server",
                border_style="bold green",
                expand=True,
            )
        )
        self.layout["output_latency_vs_output_throughput"].update(
            Panel(
                Text(),
                title="Output Latency vs Output Throughput of Server",
                border_style="bold blue",
                expand=True,
            )
        )

    def reset_panels(self):
        """Reset the histogram and live panel."""
        self.layout["input_throughput"].update(
            self._create_empty_panel("Input Throughput", "blue")
        )
        self.layout["input_latency"].update(
            self._create_empty_panel("Input Latency", "yellow")
        )
        self.layout["output_throughput"].update(
            self._create_empty_panel("Output Throughput", "blue")
        )
        self.layout["output_latency"].update(
            self._create_empty_panel("Output Latency", "yellow")
        )
        self.layout["input_histogram"].update(
            self._create_empty_panel("Input Latency Histogram", "bold green")
        )
        self.layout["output_histogram"].update(
            self._create_empty_panel("Output Latency Histogram", "bold blue")
        )

    @staticmethod
    def _create_empty_panel(title: str, border_style: str) -> Panel:
        return Panel(
            Text(),
            title=title,
            border_style=f"{border_style}",
            expand=True,
        )


Dashboard = Union[RichLiveDashboard, MinimalDashboard]


def create_dashboard(time_unit: str = "s") -> Dashboard:
    """Factory function that returns either a NoOpDashboard or RealDashboard based
    on ENABLE_UI."""
    enable_ui = os.getenv("ENABLE_UI", "true").lower() == "true"
    return RichLiveDashboard(time_unit) if enable_ui else MinimalDashboard(time_unit)
