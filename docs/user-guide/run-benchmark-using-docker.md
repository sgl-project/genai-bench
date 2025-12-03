# Docker Usage Guide

## Using Pre-built Docker Image

Pull the latest docker image:

```shell
docker pull ghcr.io/moirai-internal/genai-bench:v0.0.2
```

## Building from Source

Alternatively, you can build the image locally from the [Dockerfile](https://github.com/sgl-project/genai-bench/blob/main/Dockerfile):

```shell
docker build . -f Dockerfile -t genai-bench:dev
```

## Run Benchmark with a Docker Container

### Create a Docker Network (Optional)

To avoid internet disruptions and network latency, it's recommended to run the benchmarking within the same network as the target inference server. You can always choose to use `--network host` if you prefer.

To create a bridge network in docker:

```shell
docker network create benchmark-network -d bridge
```

### Start an Inference Server

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

### Start a Benchmark Container

Next, start the genai-bench container with the same network flag.

**Example:**

First, create a dataset configuration file to properly specify the split:

**llava-config.json:**

```json
{
  "source": {
    "type": "huggingface",
    "path": "lmms-lab/llava-bench-in-the-wild",
    "huggingface_kwargs": {
      "split": "train"
    }
  },
  "prompt_column": "question",
  "image_column": "image"
}
```

Then run the benchmark with the configuration file:

```shell
docker run \
    -tid \
    --shm-size 5g \
    --ulimit nofile=65535:65535 \
    --env HF_TOKEN="<your_HF_TOKEN>" \
    --network <your-network> \
    -v <path-to-your-local-model>:/models \
    -v $(pwd)/llava-config.json:/genai-bench/llava-config.json \
    --name llama-4-scout-benchmark \
    genai-bench:dev \
    benchmark \
    --api-backend sglang \
    --api-base http://localhost:8080 \
    --api-key your_api_key \
    --api-model-name /models/meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --model-tokenizer /models/meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --task image-text-to-text \
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
    --dataset-config /genai-bench/llava-config.json
```

Note that `genai-bench` is already the entrypoint of the container, so you only need to provide the command arguments afterward.

The genai-bench runtime UI should be available through:

```shell
docker logs --follow <CONTAINER_ID>
```

You can also utilize `tmux` for additional parallelism and session control.

## Monitor benchmark using volume mount

To monitor benchmark interim results using the genai-bench container, you can leverage volume mounts along with the `--experiment-base-dir` option.

```shell
HOST_OUTPUT_DIR=$HOME/benchmark_results
CONTAINER_OUTPUT_DIR=/genai-bench/benchmark_results
docker run \
    -tid \
    --shm-size 5g \
    --ulimit nofile=65535:65535 \
    --env HF_TOKEN="<your_HF_TOKEN>" \
    --network <your-network> \
    -v <path-to-your-local-model>:/models \
    -v $HOST_OUTPUT_DIR:$CONTAINER_OUTPUT_DIR \
    -v $(pwd)/llava-config.json:/genai-bench/llava-config.json \
    --name llama-3.2-11b-benchmark \
    genai-bench:dev \
    benchmark \
    --api-backend sglang \
    --api-base http://localhost:8080 \
    --api-key your_api_key \
    --api-model-name /models/meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --model-tokenizer /models/meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --task image-text-to-text \
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
    --dataset-config /genai-bench/llava-config.json \
    --experiment-base-dir $CONTAINER_OUTPUT_DIR
```
