from typing import Any, Optional, Tuple

import numpy as np

from genai_bench.sampling.base_scenario import (
    EmbeddingDistribution,
    ReRankDistribution,
    Scenario,
    TextDistribution,
    parse_params_str,
)


class NormalDistribution(Scenario):
    """
    Normal distribution
    e.g.
    N(300,150)/(480,240)
    """

    scenario_type = TextDistribution.NORMAL
    validation_pattern = r"^N\(\d+,\d+\)\/\(\d+,\d+\)$"

    def __init__(
        self,
        mean_input_tokens: int,
        stddev_input_tokens: int,
        mean_output_tokens: int,
        stddev_output_tokens: int,
    ):
        self.mean_input_tokens = mean_input_tokens
        self.stddev_input_tokens = stddev_input_tokens
        self.mean_output_tokens = mean_output_tokens
        self.stddev_output_tokens = stddev_output_tokens

    def sample(self) -> Tuple[int, int]:
        num_input_tokens = max(
            1,
            int(np.random.normal(self.mean_input_tokens, self.stddev_input_tokens)),
        )
        num_output_tokens = max(
            2,
            int(np.random.normal(self.mean_output_tokens, self.stddev_output_tokens)),
        )
        return num_input_tokens, num_output_tokens

    def to_string(self) -> str:
        return (
            f"N({self.mean_input_tokens},{self.stddev_input_tokens})/"
            f"({self.mean_output_tokens},{self.stddev_output_tokens})"
        )

    @classmethod
    def parse(cls, params_str: str) -> "NormalDistribution":
        (mean_input, stddev_input), (mean_output, stddev_output) = parse_params_str(
            params_str
        )
        return cls(
            mean_input_tokens=mean_input,
            stddev_input_tokens=stddev_input,
            mean_output_tokens=mean_output,
            stddev_output_tokens=stddev_output,
        )


class UniformDistribution(Scenario):
    """
    Uniform distribution
    e.g.
    U(100,200)/(200,300),
    U(1000,100)
    """

    scenario_type = TextDistribution.UNIFORM
    validation_pattern = r"^U\(\d+,\d+\)(?:\/\(\d+,\d+\))?$"

    def __init__(
        self,
        max_input_tokens: int,
        max_output_tokens: int,
        min_input_tokens: Optional[int] = None,
        min_output_tokens: Optional[int] = None,
    ):
        self.min_input_tokens = min_input_tokens
        self.max_input_tokens = max_input_tokens
        self.min_output_tokens = min_output_tokens
        self.max_output_tokens = max_output_tokens

    def sample(self) -> Tuple[int, int]:
        num_input_tokens = max(
            1,
            int(np.random.uniform(self.min_input_tokens or 1, self.max_input_tokens)),
        )
        num_output_tokens = max(
            2,
            int(np.random.uniform(self.min_output_tokens or 1, self.max_output_tokens)),
        )
        return num_input_tokens, num_output_tokens

    def to_string(self) -> str:
        if self.min_input_tokens and self.min_output_tokens:
            return (
                f"U({self.min_input_tokens},{self.max_input_tokens})/"
                f"({self.min_output_tokens},{self.max_output_tokens})"
            )
        else:
            return f"U({self.max_input_tokens},{self.max_output_tokens})"

    @classmethod
    def parse(cls, params_str: str) -> "UniformDistribution":
        parsed_parts = parse_params_str(params_str)
        if len(parsed_parts) == 2:
            (min_input, max_input), (min_output, max_output) = parsed_parts
            return cls(
                min_input_tokens=min_input,
                max_input_tokens=max_input,
                min_output_tokens=min_output,
                max_output_tokens=max_output,
            )
        else:
            (
                max_input,
                max_output,
            ) = parsed_parts[0]
            return cls(
                max_input_tokens=max_input,
                max_output_tokens=max_output,
            )


class DeterministicDistribution(Scenario):
    """
    Deterministic Distribution, aka constant
    e.g.
    D(100,1000)
    """

    scenario_type = TextDistribution.DETERMINISTIC
    validation_pattern = r"^D\(\d+,\d+\)$"

    def __init__(self, num_input_tokens: int, num_output_tokens: int):
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens

    def sample(self) -> Tuple[int, int]:
        return self.num_input_tokens, self.num_output_tokens

    def to_string(self) -> str:
        return f"D({self.num_input_tokens},{self.num_output_tokens})"

    @classmethod
    def parse(cls, params_str: str) -> "DeterministicDistribution":
        num_input, num_output = parse_params_str(params_str)[0]
        return cls(
            num_input_tokens=num_input,
            num_output_tokens=num_output,
        )


class NoOpScenario(Scenario):
    """
    A scenario that effectively performs no operation. Can be used for
    placeholders in various contexts. For example, when no input/output
    token sampling is required.

    e.g.
    F
    """

    scenario_type = TextDistribution.FILE
    validation_pattern = r"F$"

    def sample(self) -> Tuple[Any, Any]:
        return None, None

    def to_string(self) -> str:
        return "F"

    @classmethod
    def parse(cls, params_str: str) -> "NoOpScenario":
        return NoOpScenario()


class EmbeddingScenario(Scenario):
    """
    A class to represent an embedding scenario
    e.g. E(tokens_per_document)
    """

    scenario_type = EmbeddingDistribution.EMBEDDING
    validation_pattern = r"^E\(\d+\)$"

    def __init__(self, tokens_per_document: int):
        self.tokens_per_document = tokens_per_document

    def sample(self) -> int:
        """Returns tokens per document"""
        return self.tokens_per_document

    def to_string(self) -> str:
        """
        Returns the embedding scenario object back in its string representation.
        For example E(1024).
        """
        return f"E({self.tokens_per_document})"

    @classmethod
    def parse(cls, params_str: str) -> "EmbeddingScenario":
        """
        Parse the embedding scenario from a string, e.g. E(1024)
        """
        tokens_per_document = int(params_str[1:-1])
        return cls(tokens_per_document=tokens_per_document)


class ReRankScenario(Scenario):
    """
    A class to represent re-rank scenario
    e.g. R(tokens_per_document,tokens_per_query)
    """

    scenario_type = ReRankDistribution.RE_RANK
    validation_pattern = r"^R\(\d+,\d+\)$"

    def __init__(self, tokens_per_document: int, tokens_per_query: int):
        self.tokens_per_document = tokens_per_document
        self.tokens_per_query = tokens_per_query

    def sample(self) -> Tuple[int, int]:
        """Returns tokens per document"""
        return self.tokens_per_document, self.tokens_per_query

    def to_string(self) -> str:
        """
        Returns the re-rank scenario object back in its string representation.
        For example R(1024,100).
        """
        return f"R({self.tokens_per_document},{self.tokens_per_query})"

    @classmethod
    def parse(cls, params_str: str) -> "ReRankScenario":
        """
        Parse the re-rank scenario from a string, e.g. R(1024,100)
        """
        tokens_per_document, tokens_per_query = parse_params_str(params_str)[0]
        return cls(
            tokens_per_document=tokens_per_document, tokens_per_query=tokens_per_query
        )
