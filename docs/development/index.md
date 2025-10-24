# Development

Welcome and thank you for your interest in contributing to genai-bench! This section is a development guide that covers everything you need to contribute to the project.

## Getting Started with Development

<div class="grid cards" markdown>

- :material-cog:{ .lg .middle } **Adding New Features**

    ---

    Learn how to add new providers and tasks

    [:octicons-arrow-right-24: Adding New Features](adding-new-features.md)

- :material-book:{ .lg .middle } **API Reference**

    ---

    Programmatic usage and integration

    [:octicons-arrow-right-24: API Reference](api-reference.md)

</div>


## Coding Style Guide

genai-bench uses python 3.11, and we adhere to [Google Python style guide](https://google.github.io/styleguide/pyguide.html).

We use `make format` to format our code using `isort` and `ruff`. The detailed configuration can be found in
[pyproject.toml](https://github.com/sgl-project/genai-bench/blob/main/pyproject.toml).

### Guidelines

- Follow PEP 8
- Use type hints
- Write docstrings for public APIs
- Keep functions focused and small
- Add tests for new features

## Pull Requests

Please follow the PR template, which will be automatically populated when you open a new [Pull Request on GitHub](https://github.com/sgl-project/genai-bench/compare).

### Code Reviews

All submissions, including submissions by project members, require a code review.
To make the review process as smooth as possible, please:

1. Keep your changes as concise as possible.
   If your pull request involves multiple unrelated changes, consider splitting it into separate pull requests.
2. Respond to all comments within a reasonable time frame.
   If a comment isn't clear,
   or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.
3. Provide constructive feedback and meaningful comments. Focus on specific improvements
   and suggestions that can enhance the code quality or functionality. Remember to
   acknowledge and respect the work the author has already put into the submission.

## Development Setup

### Prerequisites

- Python 3.11
- Git
- Make (optional but recommended)

### Clone the Repository

```bash
git clone https://github.com/sgl-project/genai-bench.git
cd genai-bench
```

### Development Environment Setup

#### `make`

genai-bench utilizes `make` for a lot of useful commands.

If your laptop doesn't have `GNU make` installed, (check this by typing `make --version` in your terminal),
you can ask our GenerativeAI's chatbot about how to install it in your system.

#### `uv`

Install uv with `make uv` or install it from the [official website](https://docs.astral.sh/uv/).
If installing from the website, create a project venv with `uv venv -p python3.11`.

Once you have `make` and `uv` installed, you can follow the command below to build genai-bench wheel:

```shell
# check out commands genai-bench supports
make help
#activate virtual env managed by uv
source .venv/bin/activate
# install dependencies
make install
```

You can utilize wheel to install genai-bench.

```shell
# build a .whl under genai-bench/dist
make build
# send the wheel to your remote machine if applies
rsync --delete -avz ~/genai-bench/dist/<.wheel> <remote-user>@<remote-ip>:<dest-addr>
```

On your remote machine, you can simply use the `pip` to install genai-bench.

```shell
pip install <dest-addr>/<.wheel>
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
└── docs/                # Documentation
```

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
make docs

# Serve docs locally
make docs-serve

# Build docs
make docs-build
```

## Questions?
- Check out the [Adding New Features](./adding-new-features.md) and [API Reference](./api-reference.md) pages for more information on the project
- Open an issue on GitHub
- Join our community discussions
- Check existing issues and PRs