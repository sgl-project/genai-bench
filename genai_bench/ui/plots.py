import math

import numpy as np
from rich.text import Text

from genai_bench.logging import init_logger

logger = init_logger(__name__)


def create_horizontal_colored_bar_chart(
    values, width=40, bin_width=0.1, max_bins=10, time_unit="s"
):
    if not values:
        logger.warning("No data for histogram.")
        return Text()

    # Determine the range of the data
    min_value, max_value = min(values), max(values)

    # Calculate the number of bins based on the bin width
    range_of_values = max_value - min_value
    bins = max(1, int(np.ceil(range_of_values / bin_width)))
    bins = min(bins, max_bins)

    # Calculate histogram bins
    hist, bin_edges = np.histogram(values, bins=bins)

    # Determine the maximum value in the histogram for scaling
    max_value_hist = max(hist)

    chart = Text()

    # Format the labels for the bins using integer or specified decimal width
    # Convert bin edges if time_unit is "ms"
    if time_unit == "ms":
        bin_labels = [
            f"{bin_edges[i] * 1000:.0f}-{bin_edges[i + 1] * 1000:.0f}ms"
            for i in range(len(bin_edges) - 1)
        ]
    else:
        bin_labels = [
            f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}s"
            for i in range(len(bin_edges) - 1)
        ]

    # Determine the width of the longest label
    max_label_width = max(len(label) for label in bin_labels)

    for i, value in enumerate(hist):
        bar_length = int((value / max_value_hist) * (width - max_label_width - 3))
        color = (
            "red"
            if value < max_value_hist * 0.33
            else "yellow"
            if value < max_value_hist * 0.66
            else "green"
        )
        bar = f"{'█' * bar_length}"
        chart.append(f"{bin_labels[i]:<{max_label_width}} | ", style="bold")
        chart.append(bar, style=color)
        chart.append(f" {value}\n")

    return chart


def create_scatter_plot(x_values, y_values, width=40, height=10, y_unit="", x_unit=""):
    if not x_values or not y_values:
        logger.warning("No data for scatter plot.")
        return Text()

    # Calculate spacing based on time unit
    # 7 spaces for seconds, 9 spaces for milliseconds to accommodate longer labels
    label_spacing = 9 if y_unit == "ms" else 7

    # Determine ranges
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Handle the case where there's no range (e.g., all values are the same)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    # Determine the precision for y-labels based on the range and height
    y_range = y_max - y_min
    y_precision = max(0, int(-1 * round(math.log10(y_range / height))))

    # Initialize the plot grid
    plot = [[" " for _ in range(width + 1)] for _ in range(height + 1)]

    for index, (x, y) in enumerate(zip(x_values, y_values, strict=False)):
        # Normalize the points
        x_pos = int((x - x_min) / (x_max - x_min) * width)
        y_pos = height - int((y - y_min) / (y_max - y_min) * height)
        plot[y_pos][x_pos] = "•"

    # Build the plot as a string with aligned axes
    chart = Text()
    last_y_label = None

    # Add y-axis labels and plot rows
    for i, row in enumerate(plot):
        # Only label every 2nd row
        if i % 2 == 0:
            y_label = round(
                y_min + (y_max - y_min) * (height - i) / height, y_precision
            )
            # Avoid printing the same label as the row above
            if y_label != last_y_label:
                y_label_str = f"{y_label:<{label_spacing}}"
                last_y_label = y_label
                if i == 0:  # Add unit to the top row
                    y_label_str = (
                        f"{y_label:<{label_spacing - len(y_unit) - 1}} "
                        f"{y_unit:<{len(y_unit)}}"
                    )
            else:
                y_label_str = " " * label_spacing  # Leave space for the label
        else:
            y_label_str = " " * label_spacing  # Leave space for the label

        # Construct the row with dots
        plot_line = "".join(plot[i]) if i < len(plot) else ""
        chart.append(Text(f"{y_label_str}|{plot_line}\n"))

    # Add x-axis line
    x_axis_line = " " * label_spacing + "-" * (width + 1) + "\n"
    chart.append(Text(x_axis_line))

    # Create x-axis labels using actual x_values
    x_labels = [" " for _ in range(width + 10)]
    last_x_pos = -1

    for x in x_values:
        # Calculate the position of this x value on the x-axis
        x_pos = int((x - x_min) / (x_max - x_min) * width)
        label = f"{round(x)}"
        label_len = len(label)

        # Ensure the label fits in the available space and doesn't overlap
        if last_x_pos == -1 or last_x_pos + label_len < x_pos:
            x_labels[x_pos : x_pos + label_len] = list(label)
            last_x_pos = x_pos

    # Add unit to the rightmost position, on the same line
    x_label_line = (
        " " * label_spacing + "".join(x_labels) + f"{x_unit:<{len(x_unit)}}\n"
    )
    chart.append(Text(x_label_line))

    return chart
