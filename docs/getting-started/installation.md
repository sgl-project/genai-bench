# Installation Guide

This guide covers all the ways to install GenAI Bench, from simple PyPI installation to full development setup.

## Installation Methods

### Method 1: PyPI Installation (Recommended)

The simplest way to install GenAI Bench:

```bash
pip install genai-bench
```

### Method 2: Development Installation

For development or to use the latest features:

1. Please make sure you have Python 3.10-3.13 installed. You can check out online how to set it up.
2. Use the virtual environment from uv. You can install it with: 

```shell
make uv
```

Activate the virtual environment to ensure the dev environment is correctly set up:

```shell
source .venv/bin/activate
```

3. Install the Project in Editable Mode

If not already done, install your project in editable mode using make. This ensures that any changes you make are immediately reflected:

```shell
make install
```

### Method 3: Docker Installation

For containerized environments:

Pull the latest docker image:

```shell
docker pull ghcr.io/moirai-internal/genai-bench:v0.0.2
```


#### Building from Source


Alternatively, you can build the image locally from the [Dockerfile](https://github.com/sgl-project/genai-bench/blob/main/Dockerfile):

```shell
docker build . -f Dockerfile -t genai-bench:dev
```

## Verification

After installation, verify that GenAI Bench is working:

```bash
# Check version
genai-bench --version

# Check help
genai-bench --help

# Check benchmark command
genai-bench benchmark --help
```

## Environment Setup

### Environment Variables

Set these environment variables for optimal performance:

```bash
# For Hugging Face tokenizer downloads
export HF_TOKEN="your-huggingface-token"

# Disable torch warnings (not needed for benchmarking)
export TRANSFORMERS_VERBOSITY=error

# Optional: Set log level
export GENAI_BENCH_LOG_LEVEL=INFO
```

### API Keys

Depending on your backend, you may need API keys:

```bash
# OpenAI-compatible APIs
export OPENAI_API_KEY="your-api-key"

# Cohere API
export COHERE_API_KEY="your-cohere-key"

# OCI Cohere
export OCI_CONFIG_FILE="~/.oci/config"
```

## Troubleshooting

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python3 --version

# If you have multiple Python versions, use specific version (3.10-3.13)
python3.11 -m pip install genai-bench
```

#### Permission Issues
```bash
# Use user installation
pip install --user genai-bench

# Or use virtual environment
python3 -m venv genai-bench-env
source genai-bench-env/bin/activate
pip install genai-bench
```

#### Missing Dependencies
```bash
# Update pip
pip install --upgrade pip

# Install with all dependencies
pip install genai-bench[dev]
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/sgl-project/genai-bench/issues)
2. Search for similar problems
3. Create a new issue with:
   - Your operating system and Python version
   - Installation method used
   - Full error message
   - Steps to reproduce

## Running with CPU-Only PyTorch

When running genai-bench on GPU machines, PyTorch may install GPU-enabled wheels by default, which are significantly larger (~2GB+) than CPU-only wheels (~200MB). Since genai-bench primarily uses PyTorch for tokenization (not GPU computation), you can force CPU-only PyTorch installation to reduce download size and installation time.

### Option 1: Use `--torch-backend=cpu` flag

Add the `--torch-backend=cpu` flag when running with `uv`:

```bash
uv run -p 3.13 --torch-backend=cpu --with git+https://github.com/basetenlabs/genai-bench.git genai-bench benchmark
```

### Option 2: Use `UV_TORCH_BACKEND=cpu` environment variable

Set the environment variable:

```bash
UV_TORCH_BACKEND=cpu uv run -p 3.13 --with git+https://github.com/basetenlabs/genai-bench.git genai-bench benchmark
```

### Verification

After installation, verify CPU-only PyTorch is installed:

```bash
# Check torch version (should show +cpu suffix)
python -c "import torch; print(torch.__version__)"

# Verify CUDA is not available
python -c "import torch; print(torch.cuda.is_available())"  # Should print False
```

## Next Steps

After successful installation:

1. Read the [Task Definition Guide](task-definition.md) to understand different benchmark tasks
2. Explore the [User Guide](../user-guide/run-benchmark.md) for detailed usage
3. Check out [CLI Guidelines](cli-guidelines.md) for practical scenarios 