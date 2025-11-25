# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

genai-bench is a comprehensive token-level benchmarking tool for LLM serving systems. It uses Locust for distributed load testing and provides both CLI and live UI dashboard for monitoring benchmark progress.

## Development Setup

### Prerequisites

The project uses `uv` for fast Python package management and requires Python 3.10-3.12.

```bash
# Install uv (if not already installed)
make uv

# Activate virtual environment
source .venv/bin/activate

# Install project in editable mode
make install

# Install development dependencies (includes dev tools + multi-cloud support)
make dev
```

### Common Development Commands

```bash
# Run all tests with coverage
make test

# Run tests only for changed files (compared to origin/main)
make test_changed

# Format code (uses isort, ruff format, and ruff check --fix)
make format

# Run linters (ruff format --diff and mypy)
make lint

# Run both format and lint
make check

# Build wheel package
make build

# Clean build artifacts
make clean
```

### Running a Single Test

```bash
# Run a specific test file
uv run pytest tests/path/to/test_file.py -vv -s

# Run a specific test function
uv run pytest tests/path/to/test_file.py::test_function_name -vv -s

# Run with coverage for specific file
uv run pytest tests/path/to/test_file.py --cov --cov-config=.coveragerc -vv -s
```

## Architecture Overview

### Core Components

1. **CLI Entry Point** (`genai_bench/cli/cli.py`)
   - Main command: `genai-bench benchmark` with extensive options for API configuration, auth, experiments, and storage
   - Uses Click for command-line interface with option groups for organization
   - Additional commands: `genai-bench excel` and `genai-bench plot` for report generation

2. **User Classes** (`genai_bench/user/`)
   - Each User class corresponds to one API backend (OpenAI, Azure OpenAI, AWS Bedrock, GCP Vertex, OCI GenAI, Cohere)
   - All inherit from `BaseUser` which extends Locust's `HttpUser`
   - `supported_tasks` dictionary maps task names to function names
   - Each User handles API-specific request formatting and response parsing

3. **Scenarios** (`genai_bench/scenarios/`)
   - Define traffic patterns using string-based DSL (e.g., "N(500,100,200,50)" for normal distribution)
   - `Scenario.from_string()` factory method creates appropriate scenario objects
   - Types: TextDistribution (N/D/U), MultiModality (I/V/A), EmbeddingDistribution (E), ReRankDistribution (R), SpecialScenario (dataset)
   - Each scenario has `sample()` method for generating input/output token counts or modality specs

4. **Samplers** (`genai_bench/sampling/`)
   - `TextSampler` and `ImageSampler` extend `BaseSampler`
   - Generate `UserRequest` objects based on scenario specifications
   - Handle dataset loading and sample selection
   - Different samplers for different input modalities

5. **Metrics Collection** (`genai_bench/metrics/`)
   - `RequestLevelMetrics`: Single-request metrics (TTFT, TPOT, E2E latency, throughput)
   - `RequestMetricsCollector`: Calculates metrics from `UserResponse`
   - `AggregatedMetricsCollector`: Summarizes metrics across all requests in a run
   - Metrics sent via Locust's message passing to master node in distributed mode

6. **Authentication** (`genai_bench/auth/`)
   - Factory pattern with `UnifiedAuthFactory` for creating auth providers
   - Separate auth providers for models vs storage (cloud storage for datasets)
   - Cloud-specific implementations: OCI, AWS, Azure, GCP, GitHub, OpenAI
   - Two main interfaces: `ModelAuthProvider` and `StorageAuthProvider`

7. **Storage** (`genai_bench/storage/`)
   - Abstract `BaseStorage` interface for downloading datasets from cloud storage
   - Implementations: AWS S3, Azure Blob, GCP Cloud Storage, OCI Object Storage, GitHub
   - `StorageFactory` creates appropriate storage backend

8. **Protocol** (`genai_bench/protocol.py`)
   - Defines all request/response data classes using Pydantic
   - `UserRequest` variants: UserChatRequest, UserEmbeddingRequest, UserImageChatRequest, etc.
   - `UserResponse` contains timing info, tokens, status codes
   - `ExperimentMetadata` captures all benchmark configuration

9. **Analysis** (`genai_bench/analysis/`)
   - `ExperimentLoader`: Loads and parses experiment results from CSV/JSON
   - `excel_report.py`: Generates comprehensive Excel reports with metrics and pricing
   - `flexible_plot_report.py` and `plot_report.py`: Generate matplotlib visualizations
   - `PlotConfig`: Configures plot layouts (default 2x4 grid)

10. **UI Dashboard** (`genai_bench/ui/`)
    - Flask-based live dashboard showing real-time metrics during benchmark runs
    - Displays progress, logs, RPS, throughput, latency percentiles
    - Auto-refreshes during active benchmarks

11. **Data Loaders** (`genai_bench/data/loaders/`)
    - `DataLoaderFactory` creates appropriate loader (text or image)
    - Supports HuggingFace datasets and cloud storage paths
    - `DatasetConfig` specifies dataset source, column mappings, filters

### Key Patterns

- **Factory Pattern**: Used extensively for creating auth providers, storage backends, data loaders, scenarios
- **Registry Pattern**: Scenario subclasses auto-register via `__init_subclass__`
- **Protocol/Interface**: Pydantic models define strict contracts between components
- **Locust Integration**: BaseUser extends HttpUser, leverages Locust's distributed architecture and event system
- **Message Passing**: Metrics sent from workers to master using `environment.runner.send_message()`

### Adding New Tasks

When adding support for a new task:

1. Define request/response classes in `protocol.py`
2. Update or create a Sampler in `genai_bench/sampling/` to generate requests for the task
3. Add task to `supported_tasks` dict in relevant User class(es) and implement the handler function
4. Avoid duplicating logic for tasks that share the same endpoint

### Important Notes

- The codebase uses `make` extensively - always check `make help` for available commands
- Tests follow the source structure in `tests/` directory
- Code style follows Google Python style guide
- Use `isort` profile "black" with custom section for Locust imports
- Ruff configuration in `pyproject.toml` defines linting rules
- Type hints are enforced via mypy
- Pre-commit hooks available via `pre-commit install`

### Testing Changed Files

The `make test_changed` command is useful during development - it finds Python files changed since the merge-base with `origin/main` and runs only those tests. This speeds up the development cycle.

### Documentation

Documentation is built with MkDocs Material theme:

```bash
# Install docs dependencies
make docs

# Serve docs locally with live reload
make docs-serve

# Build docs to site/ directory
make docs-build
```

## Multi-Cloud Support

The tool supports benchmarking LLM APIs across multiple cloud providers:

- AWS (Bedrock models, S3 storage)
- Azure (OpenAI models, Blob storage)
- GCP (Vertex AI models, Cloud Storage)
- OCI (GenAI service, Object Storage)
- OpenAI (direct API)
- GitHub (for dataset storage)

Each provider has separate auth and storage implementations. Install cloud-specific dependencies with `pip install genai-bench[aws]`, `[azure]`, `[gcp]`, or `[multi-cloud]` for all.
