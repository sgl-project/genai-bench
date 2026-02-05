from genai_bench.analysis.plot_config import PlotConfigManager


def test_apply_time_unit_conversion_to_ms():
    """Test that labels are correctly converted from (s) to (ms)."""
    config_data = {
        "plots": [
            {
                "title": "TTFT vs Throughput",
                "y_label": "TTFT (s)",
                "x_label": "Throughput (tokens/s)",
                "y_fields": [
                    {"field": "stats.ttft.mean", "label": "Mean TTFT (s)"},
                    {"field": "stats.ttft.p95", "label": "P95 TTFT (s)"},
                ],
            }
        ]
    }

    converted = PlotConfigManager.apply_time_unit_conversion(config_data, "ms")

    # Check title conversion
    assert converted["plots"][0]["title"] == "TTFT vs Throughput"

    # Check y_label conversion
    assert converted["plots"][0]["y_label"] == "TTFT (ms)"

    # Check x_label (should not change as it's not a time field)
    assert converted["plots"][0]["x_label"] == "Throughput (tokens/s)"

    # Check y_fields labels
    assert converted["plots"][0]["y_fields"][0]["label"] == "Mean TTFT (ms)"
    assert converted["plots"][0]["y_fields"][1]["label"] == "P95 TTFT (ms)"


def test_apply_time_unit_conversion_to_seconds():
    """Test that labels are correctly converted from (ms) to (s)."""
    config_data = {
        "plots": [
            {
                "title": "Latency Analysis",
                "y_label": "E2E Latency (ms)",
                "x_label": "Concurrency",
                "y_fields": [
                    {"field": "stats.e2e_latency.mean", "label": "Mean Latency (ms)"},
                    {"field": "stats.e2e_latency.p99", "label": "P99 Latency (ms)"},
                ],
            }
        ]
    }

    converted = PlotConfigManager.apply_time_unit_conversion(config_data, "s")

    # Check y_label conversion
    assert converted["plots"][0]["y_label"] == "E2E Latency (s)"

    # Check y_fields labels
    assert converted["plots"][0]["y_fields"][0]["label"] == "Mean Latency (s)"
    assert converted["plots"][0]["y_fields"][1]["label"] == "P99 Latency (s)"


def test_apply_time_unit_conversion_no_change():
    """Test that no conversion happens when target unit matches source unit."""
    config_data = {
        "plots": [
            {
                "title": "TTFT Analysis",
                "y_label": "TTFT (s)",
                "y_fields": [{"field": "stats.ttft.mean", "label": "Mean TTFT (s)"}],
            }
        ]
    }

    # Convert to seconds (should be no-op)
    converted = PlotConfigManager.apply_time_unit_conversion(config_data, "s")

    # Check that labels remain unchanged
    assert converted["plots"][0]["y_label"] == "TTFT (s)"
    assert converted["plots"][0]["y_fields"][0]["label"] == "Mean TTFT (s)"


def test_apply_time_unit_conversion_invalid_unit():
    """Test that invalid time units are handled gracefully."""
    config_data = {"plots": [{"title": "Test Plot", "y_label": "TTFT (s)"}]}

    # Invalid time unit should return original config unchanged
    converted = PlotConfigManager.apply_time_unit_conversion(config_data, "invalid")

    assert converted == config_data


def test_apply_time_unit_conversion_empty_plots():
    """Test that empty plots list is handled correctly."""
    config_data = {"plots": []}

    converted = PlotConfigManager.apply_time_unit_conversion(config_data, "ms")

    assert converted == config_data


def test_apply_time_unit_conversion_missing_fields():
    """Test that missing fields are handled gracefully."""
    config_data = {
        "plots": [
            {
                "title": "Test Plot"
                # Missing y_label and y_fields
            }
        ]
    }

    converted = PlotConfigManager.apply_time_unit_conversion(config_data, "ms")

    # Should not crash and should return the config
    assert "title" in converted["plots"][0]
    assert converted["plots"][0]["title"] == "Test Plot"


def test_num_reasoning_tokens_in_available_plot_fields():
    """num_reasoning_tokens is included in plottable metrics."""
    fields = PlotConfigManager.get_available_fields()
    assert "stats.num_reasoning_tokens.mean" in fields
    assert "stats.num_reasoning_tokens.sum" in fields
