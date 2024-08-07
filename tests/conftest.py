from pathlib import Path

import pytest
from transformers import AutoTokenizer

from genai_bench.user.openai_user import OpenAIUser


@pytest.fixture(autouse=True)
def reset_openai_user_attrs():
    """
    Automatically resets OpenAIUser class attributes after each test to
    prevent state leakage.
    """
    # Store the original values of the class-level attributes
    original_host = OpenAIUser.host
    original_auth_provider = OpenAIUser.auth_provider

    # Yield to run the test
    yield

    # Reset the class attributes after the test
    OpenAIUser.host = original_host
    OpenAIUser.auth_provider = original_auth_provider


@pytest.fixture()
def mock_tokenizer_path():
    return str(Path(__file__).parent) + "/fixtures/local_bert_base_uncased"


@pytest.fixture
def mock_tokenizer(mock_tokenizer_path):
    # Load the real bert-base-uncased tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mock_tokenizer_path)
    return tokenizer
