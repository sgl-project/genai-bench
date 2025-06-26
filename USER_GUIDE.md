# GenAI Bench User Guide

<!-- TOC start -->

* [User Guide](#user-guide)
* [Getting Started](#getting-started)
* [Command Guidelines](#command-guidelines)
* [Task Definition](#task-definition)
  * [Note on legacy task values](#note-on-legacy)
* [Benchmark](#benchmark)
  * [Start a chat benchmark](#start-a-chat-benchmark)
  * [Start a vision based chat benchmark](#start-a-vision-benchmark)
  * [Start an embedding benchmark](#start-an-embedding-benchmark)
  * [Start a rerank benchmark](#start-a-rerank-benchmark)
  * [Start a benchmark against OCI cohere](#start-a-benchmark-against-oci-cohere)
  * [Monitor a benchmark](#monitor-a-benchmark)
  * [Notes on specific options](#notes-on-specific-options)
  * [Distributed Benchmark](#distributed-benchmark)
  * [Using Dataset Configurations](#using-dataset-configurations)
* [Generate an Excel sheet](#generate-an-excel-sheet)
* [Generate a 2x4 Plot](#generate-a-2x4-plot)
* [Upload Benchmark Results to OCI Object Storage](#uploading-benchmark-results-to-oci-object-storage)
* [Running Benchmark Using `genai-bench` Container](#running-benchmark-using-genai-bench-container)

<!-- TOC end -->

<!-- TOC --><a name="user-guide"></a>

## User Guide

This page contains a short user guide to install and use genai-bench.

<!-- TOC --><a name="getting-started"></a>

## Getting Started

1. Please make sure you have Python3.11 installed. You can check out online how to set it up.
2. Use the virtual environment from uv

Activate the virtual environment to ensure the dev environment is correctly set up:

```shell
make uv
source .venv/bin/activate
```

3. Install the Project in Editable Mode

If not already done, install your project in editable mode using make. This ensures that any changes you make are immediately reflected:

```shell
make install
```

4. Run a genai-bench command

```shell
genai-bench --help

genai-bench benchmark --help
```

<!-- TOC --><a name="command-guidelines"></a>

## Command Guidelines

Once you install it in your local environment, you can use `--help` to read
about what command options it supports.

```shell
genai-bench --help
```

`genai-bench` supports three commands:

```shell
Commands:
  benchmark  Run a benchmark based on user defined scenarios.
  excel      Exports the experiment results to an Excel file.
  plot       Plots the experiment(s) results based on filters and group...
```

You can also refer to [option_groups.py](genai_bench/cli/option_groups.py).

<!-- TOC --><a name="task-definition"></a>

## Task Definition

Tasks in `genai-bench` define the type of benchmark you want to run, based on the input modality (e.g., text, image) and output modality (e.g., text, embeddings). Tasks are specified using the `--task` option in the `genai-bench benchmark` command.

Each task follows the pattern:

```bash
<input_modality>-to-<output_modality>
```

Here are the currently supported tasks:

**NOTE**: Task compatibility may vary depending on the API format.

| Task Name             | Description                                                                  |
|-----------------------|------------------------------------------------------------------------------|
| `text-to-text`        | Benchmarks generating text output from text input, such as chat or QA tasks. |
| `text-to-embeddings`  | Benchmarks generating embeddings from text input, often for semantic search. |
| `image-to-text`       | Benchmarks generating text from images, such as visual question answering.   |
| `image-to-embeddings` | Benchmarks generating embeddings from images, often for image similarity.    |

<!-- TOC --><a name="how-tasks-work"></a>

#### How Tasks Work

* **Input Modality:** Defines the type of input data the task operates on, such as text or images.

* **Output Modality:** Defines the type of output the task generates, such as text or embeddings.

When you specify a task, the appropriate sampler (`TextSampler` or `ImageSampler`) and request type (`UserChatRequest`, `UserEmbeddingRequest`, etc.) are automatically selected based on the input and output modalities.

<!-- TOC --><a name="example-task-usage"></a>

#### Example Task Usage

* For a **text-to-text** task (e.g., generating a response to a text prompt, typical chat completions):

    ```bash
    genai-bench benchmark --task text-to-text ...
    ```

* For an **image-to-text** task (e.g., generating a response for an image and text interleave message):

    ```bash
    genai-bench benchmark --task image-to-text ...
    ```

* For an **image-to-embeddings** task (e.g., generating embeddings for similarity search):

    ```bash
    genai-bench benchmark --task text-to-embeddings ...
    ```

<!-- TOC --><a name="note-on-legacy"></a>

#### Note on legacy task values

For `genai-bench` versions prior to **0.1.75**, the supported task values are:

* **`chat`** → corresponds to `text-to-text`
* **`vision`** → corresponds to `image-to-text`
* **`embeddings`** → corresponds to `text-to-embeddings`

Be sure to use these mappings when working with earlier versions.

<!-- TOC --><a name="benchmark"></a>

## Benchmark

<!-- TOC --><a name="start-a-chat-benchmark"></a>

### Start a chat benchmark

**IMPORTANT**: Use `genai-bench benchmark --help` to check out each command option and how to use it.

For starter, you can try to type `genai-bench benchmark`, it will prompt the list of options you need to specify.

Below is a sample command you can use to start a benchmark. The command will connect with a server running on address
`http://localhost:8082`, using the default traffic scenario and num concurrency, and run each combination 1 minute.

```shell
# Optional. This is required when you load the tokenizer from huggingface.co with a model-id
export HUGGINGFACE_API_KEY="<your-key>"  
# HF transformers will log a warning about torch not installed, since benchmark doesn't really need torch 
# and cuda, we use this env to disable the warning
export TRANSFORMERS_VERBOSITY=error 

genai-bench benchmark --api-backend openai \
            --api-base "http://localhost:8082" \
            --api-key "your-openai-api-key" \
            --api-model-name "vllm-model" \
            --model-tokenizer "/mnt/data/models/Meta-Llama-3.1-70B-Instruct" \
            --task text-to-text \
            --max-time-per-run 15 \
            --max-requests-per-run 300 \
            --server-engine "vLLM" \
            --server-gpu-type "H100" \
            --server-version "v0.6.0" \
            --server-gpu-count 4 
```

<!-- TOC --><a name="start-a-vision-benchmark"></a>

### Start a vision based chat benchmark

**IMPORTANT**: Image auto-generation pipeline is not yet implemented in this repository, hence we will be using a huggingface dataset instead.

* **Image Datasets**: [Huggingface Llava Benchmark Images](https://huggingface.co/datasets/shenoyvvarun/llava-bench-in-the-wild)

Below is a sample command to trigger a vision benchmark task.

```shell
genai-bench benchmark \
            --api-backend openai \
            --api-key "your-openai-api-key" \
            --api-base "http://localhost:8180" \
            --api-model-name "/models/Phi-3-vision-128k-instruct" \
            --model-tokenizer "/models/Phi-3-vision-128k-instruct" \
            --task image-to-text \
            --max-time-per-run 15 \
            --max-requests-per-run 300 \
            --server-engine vLLM \
            --server-gpu-type A100-80G \
            --server-version "v0.6.0" \
            --server-gpu-count 4 \
            --traffic-scenario "I(256,256)" \
            --traffic-scenario "I(1024,1024)" \
            --num-concurrency 1 \
            --num-concurrency 8 \
            --dataset-path "shenoyvvarun/llava-bench-in-the-wild" \
            --dataset-image-column "image" \
            --dataset-prompt-column "conversations"
```

<!-- TOC --><a name="start-an-embedding-benchmark"></a>

### Start an embedding benchmark

Below is a sample command to trigger an embedding benchmark task. Note: when running an embedding benchmark, it is recommended to set `--num-concurrency` to 1.

```shell
genai-bench benchmark --api-backend openai \
            --api-base "http://172.18.0.3:8000" \
            --api-key "xxx" \
            --api-model-name "/models/e5-mistral-7b-instruct" \
            --model-tokenizer "/mnt/data/models/e5-mistral-7b-instruct" \
            --task text-to-embeddings \
            --server-engine "SGLang" \
            --max-time-per-run 15 \
            --max-requests-per-run 1500 \
            --traffic-scenario "E(64)" \
            --traffic-scenario "E(128)" \
            --traffic-scenario "E(512)" \
            --traffic-scenario "E(1024)" \
            --server-gpu-type "H100" \
            --server-version "v0.4.2" \
            --server-gpu-count 1 
```

<!-- TOC --><a name="start-a-rerank-benchmark"></a>

### Start a rerank benchmark against OCI Cohere

Below is a sample command to trigger a benchmark against cohere chat API.

```shell
genai-bench benchmark --api-backend oci-cohere \
            --config-file /home/ubuntu/.oci/config \
            --api-base "https://ppe.inference.generativeai.us-chicago-1.oci.oraclecloud.com" \
            --api-model-name "rerank-v3.5" \
            --model-tokenizer "Cohere/rerank-v3.5" \
            --server-engine "cohere-TensorRT" \
            --task text-to-rerank \
            --num-concurrency 1 \
            --server-gpu-type A100-80G \
            --server-version "1.7.0" \
            --server-gpu-count 4 \
            --max-time-per-run 15 \
            --max-requests-per-run 3 \
            --additional-request-params '{"compartmentId": "COMPARTMENTID", "endpointId": "ENDPOINTID", "servingType": "DEDICATED"}' \
            --num-workers 4
```


<!-- TOC --><a name="start-a-benchmark-against-oci-cohere"></a>

### Start a benchmark against OCI Cohere

Below is a sample command to trigger a benchmark against cohere chat API. 

```shell
genai-bench benchmark --api-backend oci-cohere \
            --config-file /home/ubuntu/.oci/config \
            --api-base "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com" \
            --api-model-name "c4ai-command-r-08-2024" \
            --model-tokenizer "/home/ubuntu/c4ai-command-r-08-2024" \
            --server-engine "vLLM" \
            --task text-to-text \
            --num-concurrency 1 \
            --server-gpu-type A100-80G \
            --server-version "command_r_082024_v1_7" \
            --server-gpu-count 4 \
            --max-time-per-run 15 \
            --max-requests-per-run 300 \
            --additional-request-params '{"compartmentId": "COMPARTMENTID", "endpointId": "ENDPOINTID", "servingType": "DEDICATED"}' \
            --num-workers 4
```

<!-- TOC --><a name="monitor-a-benchmark"></a>

### Monitor a benchmark

**IMPORTANT**: logs in genai-bench are all useful. Please keep an eye on WARNING logs when you finish one benchmark.

### Specify --traffic-scenario and --num-concurrency

**IMPORTANT**: Please use `genai-bench benchmark --help` to check out the latest default value of `--num-concurrency`
and `--traffic-scenario`.

Both options are defined as [multi-value options](https://click.palletsprojects.com/en/8.1.x/options/#multi-value-options) in click. Meaning you can pass this command multiple times. If you want to define your own `--num-concurrency` or `--traffic-scenario`, you can use

```shell
genai-bench benchmark \
            --api-backend openai \
            --task text-to-text \
            --max-time-per-run 10 \
            --max-requests-per-run 300 \
            --num-concurrency 1 --num-concurrency 2 --num-concurrency 4 \
            --num-concurrency 8 --num-concurrency 16 --num-concurrency 32 \
            --traffic-scenario "N(480,240)/(300,150)" --traffic-scenario "D(100,100)" 
```

<!-- TOC --><a name="notes-on-specific-options"></a>

### Notes on specific options

To manage each run or iteration in an experiment, genai-bench uses two parameters to control the exit logic. You can find more details in the `manage_run_time` function located in [utils.py](genai_bench/cli/utils.py). Combination of `--max-time-per-run` and `--max-requests-per-run` should save overall time of one benchmark.

For light traffic scenarios, such as D(7800,200) or lighter, we recommend the following settings:

```shell
            --max-time-per-run 10 \
            --max-requests-per-run 300 \
```

For heavier traffic scenarios, like `D(16000,200)` or `D(128000,200)`, use the following configuration:

```shell
            --max-time-per-run 30 \
            --max-requests-per-run 100 \
            --traffic-scenario "D(16000,200)" \
            --traffic-scenario "D(32000,200)" \
            --traffic-scenario "D(128000,200)" \
            --num-concurrency 1 \
            --num-concurrency 2 \
            --num-concurrency 4 \
            --num-concurrency 8 \
            --num-concurrency 16 \
            --num-concurrency 32 \
```

<!-- TOC --><a name="distributed-benchmark"></a>

### Distributed Benchmark

If you see the message below in the genai-bench logs, it indicates that a single process is insufficient to generate the desired load.

```log
CPU usage above 90%! This may constrain your throughput and may even give inconsistent response time measurements! 
```

To address this, you can increase the number of worker processes using the `--num-workers` option. For example, to spin up 4 worker processes, use:

```shell
    --num-workers 4
    --master-port 5577
```

This distributes the load across multiple processes on a single machine, improving performance and ensuring your benchmark runs smoothly.

#### Notes on Usage

1. This feature is experimental, so monitor the system’s behavior when enabling multiple workers. 
2. Recommended Limit: Do **not** set the number of workers to more than 16, as excessive worker processes can lead to resource contention and diminished performance.
3. Ensure your system has sufficient CPU and memory resources to support the desired number of workers. 
4. Adjust the number of workers based on your target load and system capacity to achieve optimal results.

<!-- TOC --><a name="using-dataset-configurations"></a>

### Using Dataset Configurations
Genai-bench supports flexible dataset configurations through two approaches:

#### Simple CLI Usage (for basic datasets)
```shell
# Local CSV file
--dataset-path /path/to/data.csv \
--dataset-prompt-column "prompt"

# HuggingFace dataset with simple options
--dataset-path squad \
--dataset-prompt-column "question"

# Local text file (default)
--dataset-path /path/to/prompts.txt
```

#### Advanced Configuration Files (for complex setups)
For advanced HuggingFace configurations, create a JSON config file:

**config.json:**
```json
{
  "source": {
    "type": "huggingface",
    "path": "ccdv/govreport-summarization",
    "huggingface_kwargs": {
      "split": "train",
      "revision": "main",
      "streaming": true
    }
  },
  "prompt_column": "report"
}
```

**Vision dataset config:**
```json
{
  "source": {
    "type": "huggingface",
    "path": "BLINK-Benchmark/BLINK",
    "huggingface_kwargs": {
      "split": "test",
      "name": "Jigsaw"
    }
  },
  "prompt_column": "question",
  "image_column": "image_1"
}
```

Then use: `--dataset-config config.json`

**Benefits of config files:**
- Access to ALL HuggingFace `load_dataset` parameters
- Reusable and version-controllable
- Support for complex configurations
- Future-proof (no CLI updates needed for new HuggingFace features)

<!-- TOC --><a name="generate-an-excel-sheet"></a>

## Generate an Excel sheet

genai-bench also provides the feature to analyze a finished benchmark. You can check out `genai-bench excel --help` to find how you can generate an `.xlsx` sheet containing a summary of your benchmark experiments.

### Sample command

```shell
genai-bench excel --experiment-folder <path-to-experiment-folder> --excel-name <name-of-the-sheet>
```

<!-- TOC --><a name="generate-a-2x3-plot"></a>

## Generate a 2x4 Plot

You can check out `genai-bench plot --help` to find how to generate a 2x3 Plot containing:

1. Output Inference Speed (tokens/s) vs Output Throughput of Server (tokens/s)
2. TTFT (s) vs Output Throughput of Server (tokens/s)
3. Mean E2E Latency (s) per Request vs RPS
4. Error Rates by HTTP Status vs Concurrency
5. Output Inference Speed per Request (tokens/s) vs Total Throughput (Input + Output) of Server (tokens/s)
6. TTFT (s) vs Total Throughput (Input + Output) of Server (tokens/s)
7. P90 E2E Latency (s) per Request vs RPS
8. P99 E2E Latency (s) per Request vs RPS

```shell
genai-bench plot --experiments-folder <path-to-experiment-folder> --group-key traffic_scenario
```

<!-- TOC --><a name="uploading-benchmark-results-to-oci-object-storage"></a>

## Uploading Benchmark Results to OCI Object Storage
GenAI Bench supports uploading benchmark results directly to OCI Object Storage. This feature is useful for:
- Storing benchmark results in a centralized location
- Sharing results with team members
- Maintaining a historical record of benchmarks
- Analyzing results across different runs

To enable result uploading, use the following options with the `benchmark` command:

```bash
genai-bench benchmark \
    --api-base "http://localhost:8082" \
    --api-key "your-openai-api-key" \
    --api-model-name "vllm-model" \
    --model-tokenizer "/mnt/data/models/Meta-Llama-3.1-70B-Instruct" \
    --task text-to-text \
    --max-time-per-run 15 \
    --max-requests-per-run 300 \
    --server-engine "vLLM" \
    --server-gpu-type "H100" \
    --server-version "v0.6.0" \
    --server-gpu-count 4 \
    --upload-results \
    --bucket "your-bucket-name"
```
By default, GenAI Bench uses OCI User Principal for authentication and authorization.
The default namespace is the current tenancy, and the default region is the current region in which the client is positioned.
You can override the namespace and region using the `--namespace` and `--region` options, respectively.
Alternatively, you can change the authentication and authorization mechanism using the `--auth` option.
The default object prefix is empty, but you can specify a prefix using the `--prefix` option.

<!-- TOC --><a name="running-benchmark-using-genai-bench-container"></a>

## Running Benchmark Using `genai-bench` Container

### Pending
We will publish image to docker hub soon. For now, you can build image from the [Dockerfile](./Dockerfile).

```shell
docker build . -f Dockerfile -t genai-bench:dev
```

To avoid internet disruptions and network latency, it's recommended to run the benchmarking within the same network as the target inference server. You can always choose to use `--network host` if you prefer.

To create a bridge network in docker:

```shell
docker network create benchmark-network -d bridge
```

Then, start the inference server using the standard Docker command with the additional flag `--network benchmark-network`.

**Example:**

```shell
docker run -itd \
    --gpus \"device=0,1,2,3\" \
    --shm-size 10g  -v /raid/models:/models \
    --ulimit nofile=65535:65535   --network benchmark-network \
    --name sglang-v0.4.7.post1-llama4-scout-tp4 \
    lmsysorg/sglang:v0.4.7.post1-cu124 \
    python3 -m sglang.launch_server \
    --model-path=/models/meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --tp 4 \
    --port=8080 \
    --host 0.0.0.0 \
    --context-length=131072
```

Next, start the genai-bench container with the same network flag.

**Example:**

```shell
docker run \
    -tid \
    --shm-size 5g \
    --ulimit nofile=65535:65535 \
    --env HUGGINGFACE_API_KEY="your_huggingface_api_key" \
    --network benchmark-network \
    -v /mnt/data/models:/models \
    --name llama-3.2-11b-benchmark \
    genai-bench:dev \
    benchmark \
    --api-backend openai \
    --api-base http://localhost:8080 \
    --api-key your_api_key \
    --api-model-name Llama-3.2-11B-Vision-Instruct \
    --model-tokenizer /models/Llama-3.2-11B-Vision-Instruct \
    --task image-to-text \
    --max-time-per-run 10 \
    --max-requests-per-run 100 \
    --server-engine "SGLang" \
    --server-gpu-type "H100" \
    --server-version "v0.4.7.post1" \
    --server-gpu-count 4 \
    --traffic-scenario "I(512,512)" \
    --traffic-scenario "I(2048,2048)" \
    --num-concurrency 1 \
    --num-concurrency 2 \
    --num-concurrency 4 \
    --dataset-path "shenoyvvarun/llava-bench-in-the-wild" \
    --dataset-image-column "image" \
    --dataset-prompt-column "conversations"
```

Note that `genai-bench` is already the entrypoint of the container, so you only need to provide the command arguments afterward.

The genai-bench runtime UI should be available through:

```shell
docker logs --follow <CONTAINER_ID>
```

You can also utilize `tmux` for additional parallelism and session control.

### Monitor benchmark using volume mount

To monitor benchmark interim results using the genai-bench container, you can leverage volume mounts along with the `--experiment-base-dir` option.

```shell
HOST_OUTPUT_DIR = $HOME/benchmark_results
CONTAINER_OUTPUT_DIR = /genai-bench/benchmark_results
docker run \
    -tid \
    --shm-size 5g \
    --ulimit nofile=65535:65535 \
    --env HUGGINGFACE_API_KEY="your_huggingface_api_key" \
    --network benchmark-network \
    -v /mnt/data/models:/models \
    -v $HOST_OUTPUT_DIR:$CONTAINER_OUTPUT_DIR \
    --name llama-3.2-11b-benchmark \
    genai-bench:dev \
    benchmark \
    --api-backend openai \
    --api-base http://localhost:8080 \
    --api-key your_api_key \
    --api-model-name Llama-3.2-11B-Vision-Instruct \
    --model-tokenizer /models/Llama-3.2-11B-Vision-Instruct \
    --task image-to-text \
    --max-time-per-run 10 \
    --max-requests-per-run 100 \
    --server-engine "SGLang" \
    --server-gpu-type "H100" \
    --server-version "v0.4.7.post1" \
    --server-gpu-count 4 \
    --traffic-scenario "I(512,512)" \
    --traffic-scenario "I(2048,2048)" \
    --num-concurrency 1 \
    --num-concurrency 2 \
    --num-concurrency 4 \
    --dataset-path "shenoyvvarun/llava-bench-in-the-wild" \
    --dataset-image-column "image" \
    --dataset-prompt-column "conversations" \
    --experiment-base-dir $CONTAINER_OUTPUT_DIR
```
