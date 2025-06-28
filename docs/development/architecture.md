# Architecture Guide

This guide provides an overview of GenAI Bench's architecture, design principles, and key components to help developers understand the system and contribute effectively.

## Overview

GenAI Bench is a comprehensive benchmarking framework for Generative AI APIs that focuses on performance, scalability, and ease of use. The architecture is designed to be modular, extensible, and capable of handling various types of AI workloads.

## Design Principles

### 1. Modularity
- **Separation of concerns**: Each component has a specific responsibility
- **Pluggable architecture**: Easy to add new backends, metrics, or tasks
- **Clean interfaces**: Well-defined APIs between components

### 2. Scalability
- **Concurrent execution**: Support for multiple users and requests
- **Distributed testing**: Ability to run across multiple machines
- **Resource efficiency**: Optimized for both small and large-scale testing

### 3. Extensibility
- **Backend agnostic**: Support for multiple API providers
- **Task flexibility**: Easy to add new task types
- **Metric customization**: Extensible metrics collection

### 4. Reliability
- **Error handling**: Robust error recovery and reporting
- **Data integrity**: Accurate metrics collection and storage
- **Reproducibility**: Consistent results across runs

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GenAI Bench                         │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface  │  Web UI  │  Configuration Management     │
├─────────────────┼──────────┼───────────────────────────────┤
│                 │          │                               │
│   Benchmark     │   Task   │    Traffic Scenario          │
│   Orchestrator  │ Manager  │      Management               │
│                 │          │                               │
├─────────────────┼──────────┼───────────────────────────────┤
│                 │          │                               │
│   User Pool     │ Dataset  │     Metrics Collection        │
│   Management    │ Loader   │      & Analysis               │
│                 │          │                               │
├─────────────────┼──────────┼───────────────────────────────┤
│                 │          │                               │
│   API Backend   │ Auth     │     Result Storage &          │
│   Abstraction   │ Manager  │      Reporting                │
│                 │          │                               │
└─────────────────┴──────────┴───────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   External APIs │
                    │ (OpenAI, Cohere,│
                    │  vLLM, etc.)    │
                    └─────────────────┘
```

## Core Components

### 1. CLI Interface (`genai_bench/cli/`)

The command-line interface is the primary entry point for users.

**Key Files:**
- `main.py`: Main CLI entry point
- `benchmark.py`: Benchmark command implementation
- `export.py`: Export functionality
- `plot.py`: Plotting commands

**Responsibilities:**
- Parse command-line arguments
- Validate configuration
- Orchestrate benchmark execution
- Handle user interactions

### 2. Benchmark Orchestrator (`genai_bench/`)

The core orchestration logic that coordinates all components.

**Key Files:**
- `benchmark.py`: Main benchmark orchestration
- `locust_benchmark.py`: Locust-based load testing

**Responsibilities:**
- Initialize and configure components
- Manage benchmark lifecycle
- Coordinate user simulation
- Handle distributed execution

### 3. Task Management (`genai_bench/`)

Handles different types of AI tasks and their execution.

**Task Types:**
- Text-to-text generation
- Text-to-embeddings
- Image-text-to-text (vision)
- Image-to-embeddings

**Responsibilities:**
- Task definition and validation
- Request formatting
- Response processing
- Task-specific metrics

### 4. User Pool Management (`genai_bench/user/`)

Simulates concurrent users making API requests.

**Key Files:**
- `base_user.py`: Base user class
- `openai_user.py`: OpenAI API user
- `cohere_user.py`: Cohere API user

**Responsibilities:**
- User simulation
- Request execution
- Error handling
- Response collection

### 5. API Backend Abstraction (`genai_bench/user/`)

Provides a unified interface for different API providers.

**Supported Backends:**
- OpenAI-compatible APIs
- Cohere API
- OCI Cohere API
- Custom backends

**Responsibilities:**
- API client management
- Request formatting
- Response parsing
- Error handling

### 6. Metrics Collection (`genai_bench/metrics/`)

Collects, processes, and analyzes performance metrics.

**Key Files:**
- `metrics.py`: Core metrics definitions
- `analysis.py`: Analysis utilities

**Metric Types:**
- Latency metrics (TTFT, E2E)
- Throughput metrics
- Token metrics
- Error rates
- Custom metrics

**Responsibilities:**
- Real-time metrics collection
- Statistical analysis
- Aggregation and summarization
- Export formatting

### 7. Dataset Management (`genai_bench/sampling/`)

Handles dataset loading and tokenization.

**Key Files:**
- `tokenizer.py`: Tokenization utilities
- `dataset_loader.py`: Dataset loading logic

**Dataset Types:**
- Built-in datasets (sonnet.txt, etc.)
- Custom text datasets
- Vision datasets
- Hugging Face datasets

**Responsibilities:**
- Dataset loading and validation
- Tokenization and preprocessing
- Sampling strategies
- Format conversion

### 8. Traffic Scenarios (`genai_bench/scenarios/`)

Defines different traffic patterns for testing.

**Scenario Types:**
- Constant load
- Burst traffic
- Ramp-up patterns
- Custom scenarios

**Responsibilities:**
- Traffic pattern definition
- User spawn rate control
- Load distribution
- Timing coordination

### 9. Authentication (`genai_bench/auth/`)

Manages authentication for different API providers.

**Key Files:**
- `auth.py`: Authentication utilities
- Provider-specific auth modules

**Responsibilities:**
- API key management
- Token refresh
- Authentication validation
- Security handling

### 10. Analysis & Reporting (`genai_bench/analysis/`)

Processes results and generates reports.

**Key Files:**
- `analysis.py`: Analysis utilities
- `plotting.py`: Visualization tools
- `export.py`: Export functionality

**Responsibilities:**
- Result processing
- Statistical analysis
- Report generation
- Visualization creation

### 11. Web UI (`genai_bench/ui/`)

Provides a web interface for monitoring and control.

**Responsibilities:**
- Real-time monitoring
- Interactive control
- Result visualization
- Configuration management

### 12. Distributed Execution (`genai_bench/distributed/`)

Enables distributed benchmarking across multiple machines.

**Responsibilities:**
- Cluster coordination
- Load distribution
- Result aggregation
- Fault tolerance

## Data Flow

### 1. Benchmark Initialization

```
Configuration → Validation → Component Setup → Dataset Loading
```

### 2. Benchmark Execution

```
User Spawn → Request Generation → API Calls → Response Collection → Metrics Recording
```

### 3. Result Processing

```
Raw Metrics → Aggregation → Analysis → Report Generation → Export
```

## Key Design Patterns

### 1. Strategy Pattern

Used for pluggable backends and metrics:

```python
class APIBackend(ABC):
    @abstractmethod
    def make_request(self, request: Request) -> Response:
        pass

class OpenAIBackend(APIBackend):
    def make_request(self, request: Request) -> Response:
        # OpenAI-specific implementation
        pass
```

### 2. Observer Pattern

Used for metrics collection:

```python
class MetricsCollector:
    def __init__(self):
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
    
    def notify(self, event):
        for observer in self.observers:
            observer.handle(event)
```

### 3. Factory Pattern

Used for creating users and backends:

```python
class UserFactory:
    @staticmethod
    def create_user(backend_type: str, config: Config) -> BaseUser:
        if backend_type == "openai":
            return OpenAIUser(config)
        elif backend_type == "cohere":
            return CohereUser(config)
        # ...
```

### 4. Command Pattern

Used for CLI command handling:

```python
class BenchmarkCommand:
    def __init__(self, args):
        self.args = args
    
    def execute(self):
        # Execute benchmark logic
        pass
```

## Configuration Management

### Configuration Sources

1. **Command-line arguments**: Highest priority
2. **Environment variables**: Medium priority
3. **Configuration files**: Lowest priority
4. **Defaults**: Fallback values

### Configuration Flow

```
CLI Args → Env Vars → Config Files → Defaults → Validation → Final Config
```

## Error Handling Strategy

### Error Types

1. **Configuration errors**: Invalid settings
2. **Network errors**: API connectivity issues
3. **Authentication errors**: Invalid credentials
4. **Rate limiting**: API quota exceeded
5. **Timeout errors**: Request timeouts
6. **Data errors**: Invalid datasets or responses

### Error Handling Approach

1. **Graceful degradation**: Continue when possible
2. **Detailed logging**: Comprehensive error information
3. **User feedback**: Clear error messages
4. **Recovery mechanisms**: Retry logic and fallbacks

## Performance Considerations

### Optimization Areas

1. **Concurrent execution**: Async/await patterns
2. **Memory management**: Efficient data structures
3. **Network optimization**: Connection pooling
4. **CPU optimization**: Efficient algorithms
5. **I/O optimization**: Batched operations

### Monitoring

1. **Resource usage**: CPU, memory, network
2. **Performance metrics**: Latency, throughput
3. **Error rates**: Success/failure ratios
4. **System health**: Component status

## Security Considerations

### Security Measures

1. **API key protection**: Secure storage and transmission
2. **Input validation**: Sanitize all inputs
3. **Network security**: HTTPS/TLS enforcement
4. **Access control**: Authentication and authorization
5. **Data privacy**: Secure data handling

## Testing Architecture

### Test Types

1. **Unit tests**: Component-level testing
2. **Integration tests**: Component interaction testing
3. **End-to-end tests**: Full workflow testing
4. **Performance tests**: Benchmark validation
5. **Security tests**: Vulnerability testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── fixtures/      # Test data
└── conftest.py    # Pytest configuration
```

## Deployment Architecture

### Deployment Options

1. **Local installation**: Single machine setup
2. **Docker containers**: Containerized deployment
3. **Kubernetes**: Orchestrated deployment
4. **Cloud platforms**: Managed services

### Scalability Patterns

1. **Horizontal scaling**: Multiple instances
2. **Vertical scaling**: Resource increase
3. **Load balancing**: Request distribution
4. **Caching**: Result caching

## Extension Points

### Adding New Backends

1. Implement `BaseUser` interface
2. Add backend-specific configuration
3. Register in factory
4. Add tests and documentation

### Adding New Metrics

1. Implement metric collection logic
2. Add to metrics registry
3. Update analysis tools
4. Add visualization support

### Adding New Tasks

1. Define task interface
2. Implement request/response handling
3. Add task-specific metrics
4. Update documentation

## Future Architecture Considerations

### Planned Improvements

1. **Plugin system**: Dynamic component loading
2. **Configuration UI**: Web-based configuration
3. **Real-time streaming**: Live metrics streaming
4. **ML integration**: Automated analysis
5. **Cloud native**: Kubernetes operators

### Scalability Roadmap

1. **Microservices**: Service decomposition
2. **Event streaming**: Async communication
3. **Data pipeline**: Streaming analytics
4. **Multi-region**: Global deployment

## Best Practices for Developers

### Code Organization

1. **Single responsibility**: One purpose per module
2. **Clear interfaces**: Well-defined APIs
3. **Dependency injection**: Loose coupling
4. **Error handling**: Comprehensive coverage

### Performance

1. **Async patterns**: Non-blocking operations
2. **Resource pooling**: Efficient resource use
3. **Caching**: Appropriate caching strategies
4. **Profiling**: Regular performance analysis

### Maintainability

1. **Documentation**: Comprehensive docs
2. **Testing**: High test coverage
3. **Logging**: Structured logging
4. **Monitoring**: Observability

## Troubleshooting

### Common Issues

1. **Import errors**: Module path issues
2. **Configuration errors**: Invalid settings
3. **Network issues**: Connectivity problems
4. **Performance issues**: Resource constraints

### Debugging Tools

1. **Logging**: Debug-level logging
2. **Profiling**: Performance profiling
3. **Monitoring**: System monitoring
4. **Testing**: Isolated testing

## Next Steps

- Read the [Contributing Guide](contributing.md) for development setup
- Explore the codebase following this architectural overview
- Start with small contributions to understand the system better 