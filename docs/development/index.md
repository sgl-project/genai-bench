# Development

Welcome to the GenAI Bench development guide! This section covers everything you need to contribute to the project.

## Getting Started with Development

<div class="grid cards" markdown>

-   :material-source-pull:{ .lg .middle } **Contributing**

    ---

    Learn how to contribute to GenAI Bench

    [:octicons-arrow-right-24: Contributing Guide](contributing.md)

</div>

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Make (optional but recommended)

### Clone the Repository

```bash
git clone https://github.com/sgl-project/genai-bench.git
cd genai-bench
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install in Development Mode

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
make test
```

### Code Formatting

```bash
make format
```

### Linting

```bash
make lint
```

## Project Structure

```
genai-bench/
├── genai_bench/          # Main package
│   ├── auth/            # Authentication providers
│   ├── cli/             # CLI implementation
│   ├── metrics/         # Metrics collection
│   ├── storage/         # Storage providers
│   └── user/            # User implementations
├── tests/               # Test suite
├── docs/                # Documentation
└── examples/            # Example configurations
```

## Key Components

### Authentication System
- Unified factory for creating auth providers
- Support for multiple cloud providers
- Extensible architecture for new providers

### Storage System
- Abstract base class for storage providers
- Implementations for AWS S3, Azure Blob, GCP Cloud Storage, etc.
- Consistent interface across providers

### CLI Architecture
- Click-based command structure
- Modular option groups
- Comprehensive validation

## Adding New Features

### Adding a New Model Provider

1. Create auth provider in `genai_bench/auth/`
2. Create user class in `genai_bench/user/`
3. Update `UnifiedAuthFactory`
4. Add validation in `cli/validation.py`
5. Write tests

### Adding a New Storage Provider

1. Create storage auth in `genai_bench/auth/`
2. Create storage implementation in `genai_bench/storage/`
3. Update `StorageFactory`
4. Write tests

## Testing

We use pytest for testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/auth/test_unified_factory.py

# Run with coverage
pytest --cov=genai_bench

# Run specific test
pytest -k "test_openai_auth"
```

## Documentation

Documentation uses MkDocs Material:

```bash
# Install docs dependencies
pip install mkdocs-material

# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public APIs
- Keep functions focused and small
- Add tests for new features

## Questions?

- Open an issue on GitHub
- Join our community discussions
- Check existing issues and PRs