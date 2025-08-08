import pytest

from genai_bench.scenarios import DatasetScenario, ImageModality
from genai_bench.scenarios.base import Scenario
from genai_bench.scenarios.text import (
    DeterministicDistribution,
    EmbeddingScenario,
    NormalDistribution,
    ReRankScenario,
    UniformDistribution,
)


def test_scenario_from_string_normal():
    scenario = Scenario.from_string("N(10,2)/(20,5)")
    assert isinstance(scenario, NormalDistribution)
    assert scenario.mean_input_tokens == 10
    assert scenario.stddev_input_tokens == 2
    assert scenario.mean_output_tokens == 20
    assert scenario.stddev_output_tokens == 5


def test_scenario_from_string_uniform():
    scenario = Scenario.from_string("U(5,15)/(10,25)")
    assert isinstance(scenario, UniformDistribution)
    assert scenario.min_input_tokens == 5
    assert scenario.max_input_tokens == 15
    assert scenario.min_output_tokens == 10
    assert scenario.max_output_tokens == 25

    scenario = Scenario.from_string("U(10,20)")
    assert isinstance(scenario, UniformDistribution)
    assert scenario.max_input_tokens == 10
    assert scenario.max_output_tokens == 20


def test_scenario_from_string_deterministic():
    scenario = Scenario.from_string("D(100,200)")
    assert isinstance(scenario, DeterministicDistribution)
    assert scenario.num_input_tokens == 100
    assert scenario.num_output_tokens == 200

    scenario = DeterministicDistribution.from_string("D(100,200)")
    assert isinstance(scenario, DeterministicDistribution)
    assert scenario.num_input_tokens == 100
    assert scenario.num_output_tokens == 200


def test_scenario_from_string_embedding():
    scenario = Scenario.from_string("E(1024)")
    assert isinstance(scenario, EmbeddingScenario)
    assert scenario.tokens_per_document == 1024


def test_scenario_sample_embedding():
    scenario = EmbeddingScenario(tokens_per_document=1024)
    tokens = scenario.sample()
    assert tokens == 1024


def test_scenario_from_string_image():
    scenario = Scenario.from_string("I(256,256)")
    assert isinstance(scenario, ImageModality)
    assert scenario.num_input_dimension_width == 256
    assert scenario.num_input_dimension_height == 256
    assert scenario.num_input_images == 1


def test_scenario_from_string_multipe_image():
    scenario = Scenario.from_string("I(2048,2048,2)")
    assert isinstance(scenario, ImageModality)
    assert scenario.num_input_dimension_width == 2048
    assert scenario.num_input_dimension_height == 2048
    assert scenario.num_input_images == 2


def test_scenario_from_string_invalid():
    # Invalid distribution X
    with pytest.raises(ValueError):
        Scenario.from_string("X(10,20)/(30,40)")

    # "/" missing
    with pytest.raises(ValueError):
        Scenario.from_string("N(10,20),(30,40)")

    # "-" in params
    with pytest.raises(ValueError):
        Scenario.from_string("U(-5,15)/(-10,25)")

    # extra space in params
    with pytest.raises(ValueError):
        Scenario.from_string("I(1024, 1024)")

    # not F(NaN,NaN)
    with pytest.raises(ValueError):
        Scenario.from_string("F(10,20)")

    # embedding
    with pytest.raises(ValueError):
        Scenario.from_string("E1024")

    # not valid type
    with pytest.raises(ValueError):
        Scenario.from_string("A(10,)")


def test_scenario_sample_normal():
    scenario = NormalDistribution(
        mean_input_tokens=10,
        stddev_input_tokens=200,
        mean_output_tokens=20,
        stddev_output_tokens=500,
    )
    sample = scenario.sample()

    assert len(sample) == 2
    num_input_tokens, num_output_tokens = sample
    assert num_input_tokens >= 1
    assert num_output_tokens >= 2


def test_scenario_sample_uniform():
    scenario = UniformDistribution(
        min_input_tokens=5,
        max_input_tokens=15,
        min_output_tokens=10,
        max_output_tokens=25,
    )
    sample = scenario.sample()
    assert len(sample) == 2
    num_input_tokens, num_output_tokens = sample
    assert all(isinstance(x, int) for x in sample)
    assert num_input_tokens >= 1
    assert num_output_tokens >= 2

    scenario = UniformDistribution(
        min_input_tokens=5,
        max_input_tokens=15,
        min_output_tokens=1,
        max_output_tokens=25,
    )
    sample = scenario.sample()
    assert len(sample) == 2
    num_input_tokens, num_output_tokens = sample
    assert all(isinstance(x, int) for x in sample)
    assert num_input_tokens >= 1
    assert num_output_tokens >= 2


def test_scenario_sample_deterministic():
    scenario = DeterministicDistribution(
        num_input_tokens=100,
        num_output_tokens=200,
    )
    sample = scenario.sample()
    assert sample == (100, 200)


def test_scenario_sample_image():
    scenario = ImageModality(
        num_input_dimension_width=100,
        num_input_dimension_height=200,
        max_output_token=1000,
    )
    sample = scenario.sample()
    assert sample == ((100, 200), 1, 1000)


def test_scenario_sample_multiple_image():
    scenario = ImageModality(
        num_input_dimension_width=100,
        num_input_dimension_height=200,
        num_input_images=5,
    )
    sample = scenario.sample()
    assert sample == ((100, 200), 5, None)


def test_scenario_to_string():
    scenario = NormalDistribution(
        mean_input_tokens=10,
        stddev_input_tokens=2,
        mean_output_tokens=20,
        stddev_output_tokens=5,
    )
    assert scenario.to_string() == "N(10,2)/(20,5)"

    scenario = UniformDistribution(
        min_input_tokens=5,
        max_input_tokens=15,
        min_output_tokens=10,
        max_output_tokens=25,
    )
    assert scenario.to_string() == "U(5,15)/(10,25)"

    scenario = UniformDistribution(
        max_input_tokens=10,
        max_output_tokens=20,
    )
    assert scenario.to_string() == "U(10,20)"

    scenario = Scenario.from_string("U(10,20)")
    assert scenario.to_string() == "U(10,20)"

    scenario = DeterministicDistribution(
        num_input_tokens=100,
        num_output_tokens=200,
    )
    assert scenario.to_string() == "D(100,200)"

    scenario = ImageModality(
        num_input_dimension_width=100,
        num_input_dimension_height=200,
    )
    assert scenario.to_string() == "I(100,200)"

    scenario = ImageModality(
        num_input_dimension_width=100,
        num_input_dimension_height=200,
        num_input_images=3,
    )
    assert scenario.to_string() == "I(100,200,3)"

    scenario = EmbeddingScenario(tokens_per_document=1024)
    assert scenario.to_string() == "E(1024)"

    scenario = ReRankScenario(tokens_per_document=1024, tokens_per_query=100)
    assert scenario.to_string() == "R(1024,100)"


def test_dataset_scenario_from_string():
    scenario = Scenario.from_string("dataset")
    assert isinstance(scenario, DatasetScenario)


def test_dataset_scenario_sample():
    scenario = DatasetScenario()
    with pytest.raises(NotImplementedError):
        _ = scenario.sample()


def test_dataset_scenario_to_string():
    scenario = DatasetScenario()
    assert scenario.to_string() == "dataset"


def test_rerank_from_string():
    scenario = ReRankScenario.from_string("R(1024,100)")
    max_tokens_per_doc, max_tokens_per_query = scenario.sample()
    assert isinstance(scenario, ReRankScenario)
    assert max_tokens_per_doc == 1024
    assert max_tokens_per_query == 100

    # Fails validation
    with pytest.raises(ValueError, match="Invalid scenario string"):
        ReRankScenario.from_string("E(1024,100)")
