# Contribution Guideline

Welcome and thank you for your interest in contributing to genai-bench.

## Coding Style Guide

genai-bench uses python 3.11, and we adhere to [Google Python style guide](https://google.github.io/styleguide/pyguide.html).

We use `make format` to format our code using `isort` and `ruff`. The detailed configuration can be found in
[pyproject.toml](https://github.com/sgl-project/genai-bench/blob/main/pyproject.toml).

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


## Setup Development Environment

### `make`

genai-bench utilizes `make` for a lot of useful commands.

If your laptop doesn't have `GNU make` installed, (check this by typing `make --version` in your terminal),
you can ask our GenerativeAI's chatbot about how to install it in your system.

### `uv`

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

