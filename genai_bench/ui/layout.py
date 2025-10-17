from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


def create_layout():
    layout = Layout()

    # Split the layout into the two main rows
    layout.split_column(
        Layout(name="row1", size=3),
        Layout(name="row2", ratio=1),
        Layout(name="row3", ratio=1),
        Layout(name="logs", size=15),
    )

    # Split row1 into two columns for Total Progress and Benchmark Progress
    layout["row1"].split_row(
        Layout(name="total_progress"), Layout(name="benchmark_progress")
    )

    # Split row2 into two columns for input and output
    layout["row2"].split_row(Layout(name="input"), Layout(name="output"))

    layout["row3"].split_row(
        Layout(name="ttft_vs_input_throughput", ratio=1),
        Layout(name="output_latency_vs_output_throughput", ratio=1),
    )

    layout["input"].split_row(
        Layout(name="input_side"), Layout(name="input_histogram", ratio=2)
    )

    layout["input_side"].split(
        Layout(name="input_throughput"), Layout(name="input_latency")
    )

    layout["output"].split_row(
        Layout(name="output_side"), Layout(name="output_histogram", ratio=2)
    )

    layout["output_side"].split(
        Layout(name="output_throughput"), Layout(name="output_latency")
    )

    return layout


def create_metric_panel(
    title, latency_data, throughput_data, metrics_time_unit: str = "s"
):
    # Convert latency values if needed
    if metrics_time_unit == "ms":
        # Convert seconds to milliseconds
        latency_values = {
            key: value * 1000 if value is not None else value
            for key, value in latency_data.items()
        }
        time_unit_label = "ms"
    else:
        latency_values = latency_data
        time_unit_label = "s"

    latency_table = Table.grid(expand=True)
    latency_table.add_column(justify="left")
    latency_table.add_row(
        Text.from_markup(
            f"[yellow]Avg:[/yellow] {latency_values['mean']:.2f} {time_unit_label}\n"
            f"Min: {latency_values['min']:.2f} {time_unit_label}\n"
            f"Max: {latency_values['max']:.2f} {time_unit_label}\n"
            f"[blue]P50:[/blue] {latency_values['p50']:.2f} {time_unit_label}\n"
            f"[magenta]P90:[/magenta] {latency_values['p90']:.2f} {time_unit_label}\n"
            f"[green]P99:[/green] {latency_values['p99']:.2f} {time_unit_label}",
            justify="left",
        )
    )

    throughput_table = Table.grid(expand=True)
    throughput_table.add_column(justify="left")
    throughput_table.add_row(
        Text.from_markup(
            f"[yellow]Avg:[/yellow] {throughput_data['mean']:.2f} tokens/sec\n"
            f"Min: {throughput_data['min']:.2f} tokens/sec\n"
            f"Max: {throughput_data['max']:.2f} tokens/sec",
            justify="left",
        )
    )

    latency_panel = Panel(
        latency_table, title=f"{title} Latency", border_style="yellow"
    )
    throughput_panel = Panel(
        throughput_table, title=f"{title} Throughput", border_style="blue"
    )

    return latency_panel, throughput_panel


def create_progress_bars():
    total_progress = _create_progress_bar()
    benchmark_progress = _create_progress_bar()
    total_progress_task = total_progress.add_task("Total Progress", total=100)
    return total_progress, benchmark_progress, total_progress_task


def _create_progress_bar():
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
    )


def update_progress(layout, total_progress, benchmark_progress):
    layout["total_progress"].update(
        Panel(total_progress, title="Total Progress", border_style="magenta")
    )
    layout["benchmark_progress"].update(
        Panel(
            benchmark_progress,
            title="Current Run Progress",
            border_style="cyan",
        )
    )
