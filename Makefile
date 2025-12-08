# Define the directory containing the source code
SRC_DIR := ./genai_bench
TEST_DIR := ./tests

# Define the registry and image tagging
# TODO(slin): replace with public docker registry
REGISTRY     ?= <secret>.ocir.io/<secret>/genai
TAG          ?= $(GIT_TAG)
ARCH         ?= linux/amd64
IMAGE        ?= $(REGISTRY)/genai-bench:$(TAG)

# Git version and commit information for build
GIT_TAG ?= $(shell git describe --tags --dirty --always)
LD_FLAGS += -X 'main.GitVersion=$(GIT_TAG)'
LD_FLAGS += -X 'main.GitCommit=$(shell git rev-parse HEAD)'

# Determine Docker build command (use nerdctl if available)
DOCKER_BUILD_CMD := $(shell command -v nerdctl >/dev/null 2>&1 && nerdctl info >/dev/null 2>&1 && echo nerdctl || echo docker)

# Python command configurations
PYTHON_CMD ?= python3
PIP_CMD ?= pip3

# Local binary installation path
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

.PHONY: all
all: test lint

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: install
install: ## Install project dependencies.
	uv pip install --editable .

.PHONY: dev
dev: ## Install development dependencies.
	uv pip install ".[dev,multi-cloud]"

.PHONY: test
test: ## Run tests.
	uv run pytest $(TEST_DIR) --cov --cov-config=.coveragerc -vv -s

.PHONY: test_changed
test_changed: ## Run tests only for changed files with coverage.
	@MERGEBASE=$$(git merge-base origin/main HEAD); \
	if ! git diff --diff-filter=ACM --quiet --exit-code $$MERGEBASE -- '*.py' '*.pyi' &>/dev/null; then \
		changed_files=$$(git diff --name-only --diff-filter=ACM $$MERGEBASE -- '*.py' '*.pyi'); \
		echo "Changed files: $$changed_files"; \
		uv run pytest $$changed_files --cov-config=.coveragerc -vv -s; \
	else \
		echo "No Python files have changed."; \
	fi

.PHONY: clean
clean: ## Remove build artifacts.
	rm -rf build dist *.egg-info .pytest_cache .coverage

.PHONY: format
format: ## Format code using ruff.
	uv run isort $(SRC_DIR) $(TEST_DIR)
	uv run ruff format $(SRC_DIR) $(TEST_DIR); uv run  ruff check --fix $(SRC_DIR) $(TEST_DIR)

.PHONY: lint
lint: ## Run linters using ruff.
	uv run ruff format --diff $(SRC_DIR) $(TEST_DIR)
	uv run mypy $(SRC_DIR)

.PHONY: check
check: format lint ## Run format and lint.

##@ Documentation

.PHONY: docs
docs: ## Install documentation dependencies.
	uv pip install ".[docs]"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally with live reload.
	uv run mkdocs serve -f docs/.config/mkdocs.yml

.PHONY: docs-build
docs-build: ## Build documentation to site/ directory.
	uv run mkdocs build -f docs/.config/mkdocs.yml

.PHONY: docs-deploy
docs-deploy: ## Deploy documentation to GitHub Pages.
	uv run mkdocs gh-deploy --config-file docs/.config/mkdocs-gh-pages.yml

##@ Build

.PHONY: build
build: ## Build the application.
	uv build

.PHONY: build-image
build-image: ## Build Docker image.
	$(DOCKER_BUILD_CMD) build --platform=$(ARCH) -t $(IMAGE) .

.PHONY: push-image
push-image: ## Push Docker image to registry.
	$(DOCKER_BUILD_CMD) push $(IMAGE)

##@ Dependencies

.PHONY: pipx
pipx: ## Install pipx package manager.
	$(PYTHON_CMD) -m pip install -U pipx

.PHONY: uv
uv: pipx ## Install uv and make a virtual env with
	$(PYTHON_CMD) -m pipx ensurepath; $(PYTHON_CMD) -m pipx install uv; uv venv -p $(PYTHON_CMD)
