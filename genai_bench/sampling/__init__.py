"""Import subclasses here to init registry in base class."""

from genai_bench.sampling.base_sampler import Sampler
from genai_bench.sampling.base_scenario import Scenario
from genai_bench.sampling.image_sampler import ImageSampler
from genai_bench.sampling.multi_modality_scenario import ImageModality
from genai_bench.sampling.text_sampler import TextSampler
from genai_bench.sampling.text_scenario import (
    DeterministicDistribution,
    EmbeddingScenario,
    NoOpScenario,
    NormalDistribution,
    UniformDistribution,
)
