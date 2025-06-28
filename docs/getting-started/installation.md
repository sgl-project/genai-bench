# Installation Guide

This guide covers all the ways to install GenAI Bench, from simple PyPI installation to full development setup.

## System Requirements

### Python Version
- **Required**: Python 3.11 or 3.12
- **Recommended**: Python 3.12

### Operating Systems
- **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+
- **macOS**: 10.15+ (Catalina)
- **Windows**: Windows 10+ (with WSL2 recommended)

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB+ RAM, 4+ CPU cores
- **For large benchmarks**: 16GB+ RAM, 8+ CPU cores

## Installation Methods

### Method 1: PyPI Installation (Recommended)

The simplest way to install GenAI Bench:

```bash
pip install genai-bench
```

For a specific version:

```bash
pip install genai-bench==0.1.75
```

### Method 2: Development Installation

For development or to use the latest features:

#### Prerequisites

1. **Install Python 3.11+**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-pip
   
   # macOS (using Homebrew)
   brew install python@3.11
   
   # Windows (download from python.org)
   ```

2. **Install uv (recommended package manager)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

#### Installation Steps

```bash
# Clone the repository
git clone https://github.com/sgl-project/genai-bench.git
cd genai-bench

# Create virtual environment and install dependencies
make uv

# Activate virtual environment
source .venv/bin/activate

# Install in editable mode
make install
```

### Method 3: Docker Installation

For containerized environments:

```bash
# Pull the official image
docker pull ghcr.io/sgl-project/genai-bench:latest

# Run a benchmark
docker run -it --rm ghcr.io/sgl-project/genai-bench:latest \
    genai-bench benchmark --help
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

# If you have multiple Python versions, use specific version
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

#### Network Issues
```bash
# Use alternative PyPI mirrors
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ genai-bench

# Or use conda
conda install -c conda-forge genai-bench
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

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](quick-start.md) to run your first benchmark
2. Explore the [User Guide](../user-guide/overview.md) for detailed usage
3. Check out [Examples](../examples/basic-benchmarks.md) for practical scenarios 