"""Time unit conversion utilities for genai-bench metrics."""

from typing import Any, Dict, List, Optional


class TimeUnitConverter:
    """Utility class for converting time metrics between different units."""

    # Fields that contain time/latency values (in seconds internally)
    LATENCY_FIELDS = {"ttft", "tpot", "e2e_latency", "output_latency"}

    # Stats fields that contain time statistics
    STATS_KEYS = {
        "min",
        "max",
        "mean",
        "stddev",
        "sum",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    }

    @staticmethod
    def convert_value(
        value: Optional[float], from_unit: str, to_unit: str
    ) -> Optional[float]:
        """
        Convert a time value between units.

        Args:
            value: The time value to convert (can be None)
            from_unit: Source unit ('s' or 'ms')
            to_unit: Target unit ('s' or 'ms')

        Returns:
            Converted value or None if input was None
        """
        if value is None:
            return None

        if from_unit == to_unit:
            return value

        if from_unit == "s" and to_unit == "ms":
            return value * 1000
        elif from_unit == "ms" and to_unit == "s":
            return value / 1000
        else:
            raise ValueError(f"Unsupported unit conversion: {from_unit} -> {to_unit}")

    @classmethod
    def convert_metrics_dict(
        cls, metrics_dict: Dict[str, Any], to_unit: str
    ) -> Dict[str, Any]:
        """
        Convert all time values in a metrics dictionary.

        Args:
            metrics_dict: Dictionary containing metrics data
            to_unit: Target unit ('s' or 'ms')

        Returns:
            New dictionary with converted time values
        """
        if to_unit == "s":
            return metrics_dict  # No conversion needed

        converted = metrics_dict.copy()

        # Convert direct latency fields
        for field in cls.LATENCY_FIELDS:
            if field in converted and converted[field] is not None:
                converted[field] = cls.convert_value(converted[field], "s", to_unit)

        # Convert stats nested fields
        if "stats" in converted and isinstance(converted["stats"], dict):
            converted["stats"] = converted["stats"].copy()
            for field in cls.LATENCY_FIELDS:
                if field in converted["stats"] and isinstance(
                    converted["stats"][field], dict
                ):
                    stats_obj = converted["stats"][field].copy()
                    for stat_key in cls.STATS_KEYS:
                        if stat_key in stats_obj:
                            stats_obj[stat_key] = cls.convert_value(
                                stats_obj[stat_key], "s", to_unit
                            )
                    converted["stats"][field] = stats_obj

        return converted

    @classmethod
    def convert_metrics_list(
        cls, metrics_list: List[Dict[str, Any]], to_unit: str
    ) -> List[Dict[str, Any]]:
        """
        Convert time values in a list of metrics dictionaries.

        Args:
            metrics_list: List of dictionaries containing metrics data
            to_unit: Target unit ('s' or 'ms')

        Returns:
            New list with converted time values
        """
        return [
            cls.convert_metrics_dict(metrics_dict, to_unit)
            for metrics_dict in metrics_list
        ]

    @staticmethod
    def get_unit_label(base_label: str, unit: str) -> str:
        """
        Update a label with the appropriate time unit.

        Args:
            base_label: Original label (may contain '(s)' or '(ms)' or similar)
            unit: Target unit ('s' or 'ms')

        Returns:
            Updated label with correct unit notation
        """
        if unit == "ms":
            # Replace various forms of seconds notation with milliseconds
            label = base_label.replace(" (s)", " (ms)")
            label = label.replace("(s)", "(ms)")
            label = label.replace(" per Request (seconds)", " per Request (ms)")
            label = label.replace(
                " Latency per Request (s)", " Latency per Request (ms)"
            )
            return label
        elif unit == "s":
            # Replace various forms of milliseconds notation with seconds
            label = base_label.replace(" (ms)", " (s)")
            label = label.replace("(ms)", "(s)")
            label = label.replace(" per Request (milliseconds)", " per Request (s)")
            label = label.replace(
                " Latency per Request (ms)", " Latency per Request (s)"
            )
            return label
        return base_label

    @staticmethod
    def validate_unit(unit: str) -> str:
        """
        Validate and normalize a time unit string.

        Args:
            unit: Time unit string

        Returns:
            Normalized unit string

        Raises:
            ValueError: If unit is not supported
        """
        unit = unit.lower().strip()
        if unit in ["s", "sec", "second", "seconds"]:
            return "s"
        elif unit in ["ms", "millisecond", "milliseconds"]:
            return "ms"
        else:
            raise ValueError(f"Unsupported time unit: {unit}. Supported units: s, ms")

    @classmethod
    def is_latency_field(cls, field_name: str) -> bool:
        """Check if a field name represents a latency metric."""
        return any(latency_field in field_name for latency_field in cls.LATENCY_FIELDS)
