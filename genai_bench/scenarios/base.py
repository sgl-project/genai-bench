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


class SpecialScenario(Enum):
    """Special, non-parametric scenario types."""

    DATASET = "dataset"


class Scenario(ABC):
    """
    An abstract base class for different scenarios based on specified
    distributions or modalities.
    """

    _registry: Dict[str, Type["Scenario"]] = {}
    scenario_type: (
        TextDistribution
        | MultiModality
        | EmbeddingDistribution
        | ReRankDistribution
        | SpecialScenario
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
        # Extract leading type token (supports multi-char tokens like "dataset")
        match = re.match(r"^([A-Za-z]+)", scenario_str)
        type_token = match.group(1) if match else scenario_str[0]
        cls.validate(scenario_str)
        scenario_class = cls._registry.get(type_token)
        assert scenario_class is not None, (
            "scenario_class should not be None at this step"
        )
        # Pass the parameter substring (if any) to parser
        params_str = scenario_str[len(type_token) :]
        return scenario_class.parse(params_str)

    @classmethod
    def validate(cls, scenario_str: str) -> bool:
        """
        Scenario string validation method
        Subclass requires validation_pattern to be defined
        """
        match = re.match(r"^([A-Za-z]+)", scenario_str)
        type_token = match.group(1) if match else scenario_str[0]
        scenario_class = cls._registry.get(type_token)
        if not scenario_class:
            supported = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Invalid scenario string '{scenario_str}'. Unknown type "
                f"'{type_token}'. Supported types: {supported}"
            )
        if not re.match(scenario_class.validation_pattern, scenario_str):
            raise ValueError(
                f"Invalid scenario string '{scenario_str}' for type '{type_token}'. "
                f"Expected to match pattern: {scenario_class.validation_pattern}"
            )
        return True


class DatasetScenario(Scenario):
    """
    A generic no-op scenario used to indicate dataset/direct sampling mode.
    It is registered in the scenario registry and created via its human-friendly
    alias in Scenario.from_string (e.g., "dataset").
    """

    scenario_type = SpecialScenario.DATASET
    validation_pattern = r"^dataset$"

    def sample(self) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "DatasetScenario has no sampling parameters; samplers should bypass token "
            "shaping in dataset mode."
        )

    def to_string(self) -> str:
        return "dataset"

    @classmethod
    def parse(cls, params_str: str) -> "Scenario":
        return cls()


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
