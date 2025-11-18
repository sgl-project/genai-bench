# BenchFlow - Inference Service & GenAI-Bench Workflow Orchestrator

BenchFlow is a lightweight CLI tool that orchestrates OpenAI-compatible inference service deployments (like vLLM) and GenAI-Bench benchmarking workflows on a single node. It supports both sequential and parallel execution modes, making it easy to run multiple benchmarking scenarios.

## Features

- 🚀 Automated inference service deployment and management
- 📊 Integrated GenAI-Bench benchmarking
- ⚡  Support for both sequential and parallel workflow execution
- 🔧 Configurable via JSON configuration files
- 🎯 Docker container orchestration
- 🔍 Comprehensive logging and error handling
- 🔄 Parameter grid support for experiment sweeps

## Installation

### Prerequisites

- Python 3.8+
- Docker
- Access to inference service and GenAI-Bench Docker images

### Image Mirroring (Oracle Internal Only)

If you need to mirror the GenAI-Bench image from Oracle's internal registry to OCIR:

1. Ensure you have access to Oracle Corporate Network
2. Login to both Docker registries:

   ```bash
   # Login to internal registry
   docker login odo-docker-signed-local.artifactory-builds.oci.oraclecorp.com

   # Login to OCIR
   docker login phx.ocir.io
   ```

3. Run the mirror script:

   ```bash
   # Make script executable
   chmod +x plugin/benchflow/scripts/mirror_bench_image.sh

   # Mirror specific version
   ./plugin/benchflow/scripts/mirror_bench_image.sh 0.1.83

   # Or use default version
   ./plugin/benchflow/scripts/mirror_bench_image.sh
   ```

### Install Dependencies

```bash
pip install -r plugin/benchflow/requirements.txt
```

## Usage

### Basic Commands

```bash
# Run benchmarks
python -m plugin.benchflow.entrypoint run CONFIG_PATH [--debug]

# Generate configurations only
python -m plugin.benchflow.entrypoint generate plugin/benchflow/configs/h100/llama3_2_90b_instruct_fp8_text_to_text.json \
    --param-grid  plugin/benchflow/configs/param_grid_h200.json \
    --output-file generated_configs/my_experiment.json
```

### Parameter Grid Experiments

BenchFlow supports parameter sweeping using a parameter grid configuration. You can either:

1. Generate configurations separately using the `generate` command:

```bash
# First generate the configurations
python -m plugin.benchflow.entrypoint generate plugin/benchflow/configs/h100/llama3_2_90b_instruct_fp8_text_to_text.json \
    --param-grid  plugin/benchflow/configs/param_grid_h200.json \
    --output-file generated_configs/my_experiment.json

# Then run the generated config
python -m plugin.benchflow.entrypoint run generated_configs/my_experiment.json
```

**NOTE**: ‼️‼️‼️Please **ALWAYS** double-check the generated config json before you run the experiments!!!

Example parameter grid (param_grid.json):

```json
{
    "workflows.service.version": ["v0.6.3.post1", "v0.5.3.post1"],
    "workflows.service.extra_args.gpu-memory-utilization": [0.8, 0.9, 0.95],
    "workflows.service.extra_args.tensor-parallel-size": [1, 2, 4],
    "workflows.bench.version": ["0.1.83", "0.1.75"]
}
```

This will:

1. Generate configurations for all combinations of parameters
2. Save the combined config to the specified output file
3. Run the workflows using the generated config

You can also use the ConfigGenerator API directly in your scripts:

```python
from plugin.benchflow.core.config_generator import ConfigGenerator

# Create generator from base config
generator = ConfigGenerator(base_config)

# Define parameter grid
param_grid = {
    "workflows.service.version": ["v0.6.3.post1", "v0.5.3.post1"],
    "workflows.service.extra_args.gpu-memory-utilization": [0.8, 0.9, 0.95]
}

# Generate and save variants
variants = generator.generate_variants(param_grid)
generator.save_combined_config(variants, "plugin/benchflow/generated_configs/combined_config.json")
```

See `plugin/benchflow/examples/generate_configs.py` for more examples.

### Configuration File Structure

Create a JSON configuration file with your workflow definitions:

```json
{
  "execution_mode": "sequential",
  "max_parallel": 2,
  "workflows": [
    {
      "name": "llama-3.2-11b",
      "service": {
        "container_name": "vllm-0.6.3.post1-llama3.2-11b",
        "image": "vllm/vllm-openai",
        "version": "v0.6.3.post1",
        "shm_size": "15g",
        "num_gpu_devices": 1,
        "port": 8080,
        "volumes": ["/mnt/data/models:/models:ro"],
        "env_vars": {
          "HUGGING_FACE_HUB_TOKEN": "<your-huggingface-token>"
        },
        "extra_args": [
          "--model=/models/Llama-3.2-11B-Vision-Instruct",
          "--served-model-name=vllm-model",
          "--tensor-parallel-size=1",
          "--max-num-seqs=32",
          "--enforce-eager",
          "--limit-mm-per-prompt=image=1",
          "--max-model-len=131072",
          "--gpu-memory-utilization=0.95"
        ]
      },
      "bench": {
        "container_name": "genai-bench-1",
        "image": "phx.ocir.io/idqj093njucb/genai-bench",
        "version": "0.1.75",
        "volumes": [
          "/mnt/data/models:/models:ro",
          "~/images:/genai-bench/images:rw",
          "/path/to/results:/results:rw"
        ],
        "env_vars": {
          "GENAI_BENCH_LOGGING_LEVEL": "INFO",
          "HUGGINGFACE_API_KEY": "<your-huggingface-token>"
        },
        "extra_args": [
          "--api-key", "your_api_key",
          "--api-model-name", "vllm-model",
          "--model-tokenizer", "/models/Llama-3.2-11B-Vision-Instruct",
          "--task", "image-to-text",
          "--max-time-per-run", "10",
          "--max-requests-per-run", "100",
          "--server-engine", "vLLM",
          "--server-gpu-type", "H100",
          "--server-version", "v0.6.2",
          "--server-gpu-count", "1",
          "--traffic-scenario", "I(512,512)",
          "--traffic-scenario", "I(2048,2048)",
          "--num-concurrency", "1",
          "--num-concurrency", "2",
          "--num-concurrency", "4",
          "--dataset-path", "./images/questions.jsonl"
        ]
      }
    }
  ]
}
```

### Execution Modes

1. **Sequential Mode**: Runs workflows one after another

   ```json
   {
     "execution_mode": "sequential",
     "workflows": [...]
   }
   ```

2. **Parallel Mode**: Runs multiple workflows concurrently

   ```json
   {
     "execution_mode": "parallel",
     "max_parallel": 2,
     "workflows": [...]
   }
   ```

## Key Components

### InferenceServiceConfig

- Manages inference service deployment settings
- Configures GPU utilization and service parameters
- Controls Docker container settings for the service
- Supports flexible command-line arguments via extra_args

### BenchConfig

- Defines benchmark execution parameters
- Sets up GenAI-Bench container configuration
- Manages load testing scenarios and concurrency levels

### WorkflowPair

- Links inference service and benchmark configurations
- Provides unique workflow identification
- Ensures coordinated execution of service and benchmark components

### ConfigGenerator

- Generates workflow configurations from parameter grids
- Supports parameter sweeping for experiments
- Handles linked parameters between service and bench configs
- Saves combined configurations for execution

## Error Handling

The orchestrator includes comprehensive error handling for:

- Docker operations (image pulling, container management)
- Network creation and cleanup
- Server health checks
- Resource cleanup on failure

## Logging

Detailed logging is provided for all operations:

- Container startup and shutdown
- Workflow execution progress
- Error conditions and cleanup operations
- Benchmark results and metrics

## Notes

- Ensure Docker daemon is running before execution
- Verify access to required Docker images
- Check GPU availability for service deployment
- Monitor system resources during parallel execution
