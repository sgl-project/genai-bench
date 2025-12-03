# Claude Code Notes for genai-bench

## Running Tests

This project uses **pytest** with a 93% coverage threshold enforced in CI.

### Quick Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run tests for a specific module
uv run pytest tests/metrics/ -v

# Run a single test file
uv run pytest tests/metrics/test_metrics.py -v

# Run with coverage report
uv run pytest tests/ --cov --cov-report=term-missing
```

### Test Structure

Tests are organized by module under `tests/`:
- `tests/metrics/` - Metrics collection and aggregation
- `tests/cli/` - CLI interface tests
- `tests/async_runner/` - Async runner tests
- `tests/user/` - API user/client tests
- `tests/auth/` - Authentication provider tests
- `tests/sampling/` - Request sampling tests
- `tests/scenarios/` - Traffic scenario tests

### Key Fixtures

Defined in `tests/conftest.py`:
- `mock_tokenizer` - Loads bert-base-uncased tokenizer from `tests/fixtures/`
- `mock_tokenizer_path` - Path to local tokenizer fixture

### Notes

- Tests use `pytest-asyncio` for async test support
- The test suite includes gevent monkey-patching for Locust compatibility
- Some tests may show numpy warnings for edge cases (inf/nan) - these are expected
