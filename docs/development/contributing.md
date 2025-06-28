# Contributing to GenAI Bench

Thank you for your interest in contributing to GenAI Bench! This guide will help you get started with development and understand our contribution process.

## Development Setup

### Prerequisites

- Python 3.11 or 3.12
- Git
- Make (optional, for using Makefile commands)

### Local Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sgl-project/genai-bench.git
   cd genai-bench
   ```

2. **Set up virtual environment**:
   ```bash
   make uv
   source .venv/bin/activate
   ```

3. **Install in editable mode**:
   ```bash
   make install
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Verify Setup

Test that everything is working:

```bash
# Run tests
make test

# Check code quality
make lint

# Run type checking
make type-check
```

## Project Structure

```
genai-bench/
├── genai_bench/           # Main package
│   ├── cli/              # Command-line interface
│   ├── metrics/          # Metrics collection and analysis
│   ├── sampling/         # Data sampling and tokenization
│   ├── scenarios/        # Traffic scenarios
│   ├── ui/               # Web UI components
│   ├── user/             # API user implementations
│   ├── auth/             # Authentication modules
│   ├── analysis/         # Result analysis tools
│   ├── distributed/      # Distributed benchmarking
│   └── oci_object_storage/ # OCI integration
├── tests/                # Test suite
├── examples/             # Example scripts and configs
├── docs/                 # Documentation
├── pyproject.toml        # Project configuration
├── Makefile              # Development commands
└── README.md             # Project overview
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

Follow our coding standards and make your changes.

### 3. Run Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_specific_module.py

# Run with coverage
make test-cov
```

### 4. Check Code Quality

```bash
# Run all quality checks
make lint

# Run specific checks
make ruff
make black
make isort
make mypy
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use isort with Black profile
- **Type hints**: Required for all public functions
- **Docstrings**: Use Google style docstrings

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code
make format

# Check formatting
make check-format
```

### Type Checking

We use MyPy for static type checking:

```bash
# Run type checking
make type-check

# Type check specific file
mypy genai_bench/your_module.py
```

### Linting

We use Ruff for linting:

```bash
# Run linter
make lint

# Fix auto-fixable issues
make lint-fix
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Use pytest as the testing framework
- Follow the naming convention: `test_*.py`
- Use descriptive test names

### Test Structure

```python
import pytest
from genai_bench.your_module import your_function


def test_your_function_basic():
    """Test basic functionality."""
    result = your_function("input")
    assert result == "expected_output"


def test_your_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        your_function("")


@pytest.fixture
def sample_data():
    """Provide test data."""
    return {"key": "value"}


def test_your_function_with_fixture(sample_data):
    """Test with fixture data."""
    result = your_function(sample_data)
    assert result is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_specific.py::test_function

# Run with coverage
pytest --cov=genai_bench

# Run integration tests
pytest tests/integration/
```

## Documentation

### Code Documentation

- Use Google style docstrings for all public functions
- Include type hints for all parameters and return values
- Document exceptions that may be raised

Example:

```python
def process_data(data: str, config: Dict[str, Any]) -> List[str]:
    """Process input data according to configuration.
    
    Args:
        data: Input string to process.
        config: Configuration dictionary.
        
    Returns:
        List of processed strings.
        
    Raises:
        ValueError: If data is empty or invalid.
        ConfigError: If configuration is invalid.
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Implementation here
    return processed_data
```

### Documentation Updates

When adding new features:

1. Update relevant documentation files in `docs/`
2. Add examples if applicable
3. Update API documentation
4. Ensure all links work correctly

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: Run `make test`
2. **Check code quality**: Run `make lint`
3. **Update documentation**: Add/update relevant docs
4. **Add tests**: Include tests for new functionality

### Pull Request Template

Use this template when creating a PR:

```markdown
## Description

Brief description of the changes.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] All existing tests pass
- [ ] I have tested the changes manually

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Documentation review** if applicable
4. **Testing verification** if needed

## Areas for Contribution

### High Priority

- **Performance improvements**: Optimize existing functionality
- **Bug fixes**: Fix reported issues
- **Documentation**: Improve existing docs or add missing sections
- **Test coverage**: Add tests for uncovered code

### Medium Priority

- **New features**: Add requested functionality
- **API improvements**: Enhance existing APIs
- **UI enhancements**: Improve web interface
- **Integration**: Add support for new backends

### Low Priority

- **Code refactoring**: Improve code structure
- **Tooling**: Enhance development tools
- **Examples**: Add more example configurations

## Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Before Asking for Help

1. **Check existing issues**: Search for similar problems
2. **Read documentation**: Check relevant docs
3. **Try debugging**: Use debug mode and logs
4. **Provide context**: Include error messages and environment details

### Creating Good Issues

When reporting bugs or requesting features:

1. **Use clear titles**: Descriptive and specific
2. **Provide context**: Environment, version, steps to reproduce
3. **Include logs**: Error messages and relevant output
4. **Add examples**: Code snippets or configuration files

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

Before a release:

1. **Update version**: Update version in `pyproject.toml`
2. **Update changelog**: Document changes
3. **Run full test suite**: Ensure all tests pass
4. **Update documentation**: Ensure docs are current
5. **Create release notes**: Summarize changes

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Respect different viewpoints

### Enforcement

- Report violations to maintainers
- Maintainers will address issues promptly
- Violations may result in temporary or permanent exclusion

## License

By contributing to GenAI Bench, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for contributing to GenAI Bench! Your contributions help make this project better for everyone.

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand the codebase
- Start with small contributions to understand the system better
- Join the community discussions for questions and ideas