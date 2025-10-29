"""Tests for text scenario implementations."""

from genai_bench.scenarios.text import (
    DeterministicDistribution,
    EmbeddingScenario,
    NormalDistribution,
    PrefixRepetitionScenario,
    ReRankScenario,
    UniformDistribution,
)


def test_normal_distribution_creation():
    """Test NormalDistribution creation and sampling."""
    scenario = NormalDistribution(
        mean_input_tokens=100,
        stddev_input_tokens=10,
        mean_output_tokens=50,
        stddev_output_tokens=5,
    )

    input_tokens, output_tokens = scenario.sample()
    assert isinstance(input_tokens, int)
    assert isinstance(output_tokens, int)
    assert input_tokens >= 1  # Should be clamped to minimum
    assert output_tokens >= 2  # Should be clamped to minimum


def test_uniform_distribution_creation():
    """Test UniformDistribution creation and sampling."""
    scenario = UniformDistribution(
        min_input_tokens=10,
        max_input_tokens=20,
        min_output_tokens=5,
        max_output_tokens=15,
    )

    input_tokens, output_tokens = scenario.sample()
    assert 10 <= input_tokens <= 20
    assert 5 <= output_tokens <= 15


def test_deterministic_distribution():
    """Test DeterministicDistribution."""
    scenario = DeterministicDistribution(
        num_input_tokens=100,
        num_output_tokens=50,
    )

    input_tokens, output_tokens = scenario.sample()
    assert input_tokens == 100
    assert output_tokens == 50


def test_embedding_scenario():
    """Test EmbeddingScenario."""
    scenario = EmbeddingScenario(tokens_per_document=1024)

    tokens = scenario.sample()
    assert tokens == 1024


def test_rerank_scenario():
    """Test ReRankScenario."""
    scenario = ReRankScenario(tokens_per_document=512, tokens_per_query=64)

    doc_tokens, query_tokens = scenario.sample()
    assert doc_tokens == 512
    assert query_tokens == 64


def test_normal_distribution_to_string():
    """Test NormalDistribution string representation."""
    scenario = NormalDistribution(
        mean_input_tokens=10,
        stddev_input_tokens=2,
        mean_output_tokens=20,
        stddev_output_tokens=5,
    )
    assert scenario.to_string() == "N(10,2)/(20,5)"


def test_uniform_distribution_to_string():
    """Test UniformDistribution string representation."""
    scenario = UniformDistribution(
        min_input_tokens=5,
        max_input_tokens=15,
        min_output_tokens=10,
        max_output_tokens=25,
    )
    assert scenario.to_string() == "U(5,15)/(10,25)"


def test_deterministic_distribution_to_string():
    """Test DeterministicDistribution string representation."""
    scenario = DeterministicDistribution(
        num_input_tokens=100,
        num_output_tokens=200,
    )
    assert scenario.to_string() == "D(100,200)"


def test_embedding_scenario_to_string():
    """Test EmbeddingScenario string representation."""
    scenario = EmbeddingScenario(tokens_per_document=1024)
    assert scenario.to_string() == "E(1024)"


def test_rerank_scenario_to_string():
    """Test ReRankScenario string representation."""
    scenario = ReRankScenario(tokens_per_document=1024, tokens_per_query=100)
    assert scenario.to_string() == "R(1024,100)"


def test_prefix_repetition_scenario():
    """Test PrefixRepetitionScenario creation and sampling."""
    scenario = PrefixRepetitionScenario(
        prefix_len=2000,
        suffix_len=500,
        output_len=200,
    )

    prefix_len, suffix_len, output_len = scenario.sample()
    assert prefix_len == 2000
    assert suffix_len == 500
    assert output_len == 200


def test_prefix_repetition_scenario_to_string():
    """Test PrefixRepetitionScenario string representation."""
    scenario = PrefixRepetitionScenario(
        prefix_len=2000,
        suffix_len=500,
        output_len=200,
    )
    assert scenario.to_string() == "P(2000,500)/200"


def test_prefix_repetition_scenario_parse():
    """Test PrefixRepetitionScenario parsing from string."""
    scenario = PrefixRepetitionScenario.parse("(2000,500)/200")
    
    assert scenario.prefix_len == 2000
    assert scenario.suffix_len == 500
    assert scenario.output_len == 200


def test_prefix_repetition_scenario_parse_different_values():
    """Test PrefixRepetitionScenario parsing with various values."""
    test_cases = [
        ("(1000,500)/100", 1000, 500, 100),
        ("(4000,1000)/200", 4000, 1000, 200),
        ("(8000,2000)/500", 8000, 2000, 500),
    ]
    
    for params_str, expected_prefix, expected_suffix, expected_output in test_cases:
        scenario = PrefixRepetitionScenario.parse(params_str)
        assert scenario.prefix_len == expected_prefix
        assert scenario.suffix_len == expected_suffix
        assert scenario.output_len == expected_output


def test_prefix_repetition_scenario_invalid_format():
    """Test PrefixRepetitionScenario parsing with invalid format."""
    import pytest
    
    invalid_formats = [
        "(2000,500)",  # Missing output
        "(2000)/200",  # Missing suffix
        "2000,500/200",  # Missing parentheses
        "(2000,500,200)",  # Wrong separator
    ]
    
    for invalid_format in invalid_formats:
        with pytest.raises(ValueError):
            PrefixRepetitionScenario.parse(invalid_format)
