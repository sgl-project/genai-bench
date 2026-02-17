# Use the official Python 3.11 slim image from Docker Hub
FROM python:3.14-slim

# Set the working directory in the container
WORKDIR /genai-bench
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    gcc \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pipx and uv using pip
RUN pip install --upgrade pip pipx hatchling wheel && \
    pipx ensurepath && \
    pipx install uv && \
    rm -rf /root/.cache

# Copy the application code to the working directory
COPY . .

# Install the project and dependencies using uv
ARG PKG_VERSION
RUN echo "Installing version:${PKG_VERSION}" && \
    uv version ${PKG_VERSION} && \
    uv pip install --system -vvv . && \
    rm -rf /root/.cache

# Clean up unnecessary files to reduce the image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/log/*

# start poetry shell to activate the virtual environment
ENTRYPOINT ["genai-bench"]
