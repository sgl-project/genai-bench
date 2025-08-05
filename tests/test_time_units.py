"""Tests for time unit conversion functionality."""

import pytest

from genai_bench.time_units import TimeUnitConverter


class TestTimeUnitConverter:
    """Test TimeUnitConverter utility class."""

    def test_basic_conversion(self):
        """Test basic time unit conversions."""
        # Seconds to milliseconds
        assert TimeUnitConverter.convert_value(1.0, "s", "ms") == 1000.0
        assert TimeUnitConverter.convert_value(1.5, "s", "ms") == 1500.0
        assert TimeUnitConverter.convert_value(0.001, "s", "ms") == 1.0

        # Milliseconds to seconds
        assert TimeUnitConverter.convert_value(1000.0, "ms", "s") == 1.0
        assert TimeUnitConverter.convert_value(1500.0, "ms", "s") == 1.5
        assert TimeUnitConverter.convert_value(1.0, "ms", "s") == 0.001

        # Same unit (no conversion)
        assert TimeUnitConverter.convert_value(5.0, "s", "s") == 5.0
        assert TimeUnitConverter.convert_value(5000.0, "ms", "ms") == 5000.0

    def test_none_values(self):
        """Test conversion with None values."""
        assert TimeUnitConverter.convert_value(None, "s", "ms") is None
        assert TimeUnitConverter.convert_value(None, "ms", "s") is None

    def test_invalid_conversion(self):
        """Test invalid unit conversion raises error."""
        with pytest.raises(ValueError):
            TimeUnitConverter.convert_value(1.0, "s", "invalid")

        with pytest.raises(ValueError):
            TimeUnitConverter.convert_value(1.0, "invalid", "s")

    def test_label_conversion(self):
        """Test label unit conversion."""
        # Basic label updates
        assert TimeUnitConverter.get_unit_label("TTFT", "ms") == "TTFT"
        assert TimeUnitConverter.get_unit_label("TTFT (s)", "ms") == "TTFT (ms)"
        assert (
            TimeUnitConverter.get_unit_label("End-to-End Latency per Request (s)", "ms")
            == "End-to-End Latency per Request (ms)"
        )

        # No change for seconds
        assert TimeUnitConverter.get_unit_label("TTFT (s)", "s") == "TTFT (s)"

    def test_metrics_dict_conversion(self):
        """Test conversion of metrics dictionary."""
        test_metrics = {
            "ttft": 1.5,
            "e2e_latency": 2.3,
            "tpot": 0.1,
            "output_latency": 1.0,
            "throughput": 100,  # Should not be converted
            "stats": {
                "ttft": {"mean": 1.5, "p90": 2.0, "p99": 2.5},
                "e2e_latency": {"mean": 2.3, "p90": 3.0, "p99": 3.5},
                "throughput": {"mean": 100, "p90": 120},  # Should not be converted
            },
        }

        converted = TimeUnitConverter.convert_metrics_dict(test_metrics, "ms")

        # Check direct fields are converted
        assert converted["ttft"] == 1500.0
        assert converted["e2e_latency"] == 2300.0
        assert converted["tpot"] == 100.0
        assert converted["output_latency"] == 1000.0

        # Check non-time field is unchanged
        assert converted["throughput"] == 100

        # Check stats are converted
        assert converted["stats"]["ttft"]["mean"] == 1500.0
        assert converted["stats"]["ttft"]["p90"] == 2000.0
        assert converted["stats"]["ttft"]["p99"] == 2500.0
        assert converted["stats"]["e2e_latency"]["mean"] == 2300.0

        # Check non-time stats are unchanged
        assert converted["stats"]["throughput"]["mean"] == 100

        # Check that no conversion happens for seconds
        converted_s = TimeUnitConverter.convert_metrics_dict(test_metrics, "s")
        assert converted_s == test_metrics

    def test_metrics_list_conversion(self):
        """Test conversion of list of metrics dictionaries."""
        test_metrics_list = [
            {"ttft": 1.0, "throughput": 50},
            {"ttft": 1.5, "throughput": 75},
            {"ttft": 2.0, "throughput": 100},
        ]

        converted = TimeUnitConverter.convert_metrics_list(test_metrics_list, "ms")

        assert len(converted) == 3
        assert converted[0]["ttft"] == 1000.0
        assert converted[1]["ttft"] == 1500.0
        assert converted[2]["ttft"] == 2000.0

        # Check non-time fields unchanged
        assert converted[0]["throughput"] == 50
        assert converted[1]["throughput"] == 75
        assert converted[2]["throughput"] == 100

    def test_unit_validation(self):
        """Test unit validation and normalization."""
        assert TimeUnitConverter.validate_unit("s") == "s"
        assert TimeUnitConverter.validate_unit("sec") == "s"
        assert TimeUnitConverter.validate_unit("second") == "s"
        assert TimeUnitConverter.validate_unit("seconds") == "s"

        assert TimeUnitConverter.validate_unit("ms") == "ms"
        assert TimeUnitConverter.validate_unit("millisecond") == "ms"
        assert TimeUnitConverter.validate_unit("milliseconds") == "ms"

        with pytest.raises(ValueError):
            TimeUnitConverter.validate_unit("invalid")

    def test_is_latency_field(self):
        """Test latency field detection."""
        assert TimeUnitConverter.is_latency_field("ttft") is True
        assert TimeUnitConverter.is_latency_field("tpot") is True
        assert TimeUnitConverter.is_latency_field("e2e_latency") is True
        assert TimeUnitConverter.is_latency_field("output_latency") is True

        assert TimeUnitConverter.is_latency_field("throughput") is False
        assert TimeUnitConverter.is_latency_field("num_tokens") is False
        assert TimeUnitConverter.is_latency_field("error_rate") is False
