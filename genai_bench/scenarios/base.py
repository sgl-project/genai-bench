import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Tuple, Type


class TextDistribution(Enum):
    """
    Distribution type for text input output sampling
    """

    NORMAL = "N"
    DETERMINISTIC = "D"
    UNIFORM = "U"
    FILE = "F"  # Special case within text distribution


class EmbeddingDistribution(Enum):
    """
    Distribution type for embedding input sampling
    """

    EMBEDDING = "E"


class ReRankDistribution(Enum):
    """
    Distribution type for re-ranking documents as per query.
    """

    RE_RANK = "R"


class MultiModality(Enum):
    """
    Enumeration for multi modality scenario sampling
    """

    IMAGE = "I"  # I(width, height)
    VIDEO = "V"
    AUDIO = "A"


class Scenario(ABC):
    """
    An abstract base class for different scenarios based on specified
    distributions or modalities.
    """

    _registry: Dict[str, Type["Scenario"]] = {}
    scenario_type: (
        TextDistribution | MultiModality | EmbeddingDistribution | ReRankDistribution
    )
    validation_pattern: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.scenario_type.value] = cls

    @abstractmethod
    def sample(self) -> Any:
        """
        Samples the number of input and output tokens based on the distribution
        and parameters.
        Or sampling different modalities specs such as
        image width and height
        """
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Converts the Scenario object back into its string representation."""
        pass

    @classmethod
    @abstractmethod
    def parse(cls, params_str: str) -> "Scenario":
        """Parses scenario str to a Scenario object."""
        pass

    @classmethod
    def from_string(cls, scenario_str: str) -> "Scenario":
        """
        Factory method that creates a Scenario object from its string
        representation.
        Determines the correct subclass to instantiate using the registered map.
        """
        cls.validate(scenario_str)
        type_identifier = scenario_str[0]
        scenario_class = cls._registry.get(type_identifier)
        assert scenario_class is not None, (
            "scenario_class should not be None at this step"
        )
        return scenario_class.parse(scenario_str[1:])

    @classmethod
    def validate(cls, scenario_str: str) -> bool:
        """
        Scenario string validation method
        Subclass requires validation_pattern to be defined
        """
        scenario_type = scenario_str[0]
        scenario_class = cls._registry.get(scenario_type)
        if not scenario_class or not bool(
            re.match(scenario_class.validation_pattern, scenario_str)
        ):
            raise ValueError(
                f"Invalid scenario string '{scenario_str}'. Should follow "
                "U(min_input_tokens,max_input_tokens)/(min_output_tokens,max_output_tokens), "  # noqa: E501
                "N(mean_input_tokens,stddev_input_tokens)/(mean_output_tokens,stddev_output_tokens), "  # noqa: E501
                "D(num_input_tokens,num_output_tokens), "
                "U(max_input_tokens,max_output_tokens) OR "
                "Multi modality scenario I(num_input_dimension_width,num_input_dimension_height,num_input_images)"  # noqa: E501
                "Embedding scenario: E(max_tokens_per_document)"
                "Re-Rank scenario: R(max_tokens_per_document,max_tokens_per_query)"
            )
        return True


def parse_params_str(
    params_str: str,
) -> List[Tuple[int, ...]]:
    """
    Parses the parameter string into tuples of integers.

    Args:
        params_str: The parameter string to parse.

    Returns:
        A list of tuples containing integers parsed from the string.
    """
    if "/" in params_str:
        parts = params_str.split("/")
        return [tuple(map(int, part[1:-1].split(","))) for part in parts]
    else:
        return [tuple(map(int, params_str[1:-1].split(",")))]
