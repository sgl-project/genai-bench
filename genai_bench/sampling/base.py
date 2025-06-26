from abc import ABC, abstractmethod
from typing import Dict, Optional, Set, Type

from transformers import PreTrainedTokenizer

from genai_bench.protocol import UserRequest
from genai_bench.scenarios.base import Scenario


class Sampler(ABC):
    """Abstract base class for samplers. Responsible for sampling image/text from
    a dataset. Refrain from adding here additional logic like parsing, encoding etc.

    This class defines the interface for all samplers and provides
    a registry for different task-specific samplers.
    """

    modality_registry: Dict[str, Type["Sampler"]] = {}
    input_modality: str
    supported_tasks: Set[str]

    def __init_subclass__(cls, **kwargs):
        """Automatically registers subclasses in the task registry."""
        super().__init_subclass__(**kwargs)
        cls.modality_registry[cls.input_modality] = cls

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: str,
        output_modality: str,
        additional_request_params: Optional[dict] = None,
        **kwargs,
    ):
        """Initializes the Sampler.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding
                text.
            model (str): The name of the model to use.
            output_modality (str): The output modality (e.g., "text" or
                "embeddings").
            additional_request_params (Optional[dict]): Additional parameters
                for the request.
            **kwargs: Additional keyword arguments.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.output_modality = output_modality
        self.additional_request_params = additional_request_params or {}
        self.get_token_length = lambda text, add_special_tokens=False: len(
            tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )
        self.use_scenario = True
        self.batch_size = 1  # Default batch size

    @abstractmethod
    def sample(self, scenario: Scenario) -> UserRequest:
        """Samples a request based on the given scenario.

        Args:
            scenario (Scenario): The scenario to use for sampling.

        Returns:
            UserRequest: A request object.
        """
        pass

    @classmethod
    def create(cls, task: str, *args, **kwargs) -> "Sampler":
        """Creates a task-specific sampler instance.

        Args:
            task (str): Task name in the format "<input>-to-<output>".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Sampler: An instance of the appropriate sampler.

        Raises:
            ValueError: If no sampler supports the specified task.
        """
        try:
            input_modality, output_modality = task.split("-to-")
        except ValueError as err:
            raise ValueError(
                f"Invalid task format: {task}. Expected '<input>-to-<output>'."
            ) from err

        # Check if the input modality is supported
        if input_modality not in cls.modality_registry:
            raise ValueError(f"No sampler supports input modality: {input_modality}")

        sampler_cls = cls.modality_registry[input_modality]
        if not sampler_cls.supports_task(input_modality, output_modality):
            raise ValueError(
                f"Sampler for {input_modality} does not support output "
                f"modality: {output_modality}"
            )

        return sampler_cls(*args, output_modality=output_modality, **kwargs)  # type: ignore[misc]

    @classmethod
    def supports_task(cls, input_modality: str, output_modality: str) -> bool:
        """Checks if the sampler supports a given input-to-output task."""
        task_name = f"{input_modality}-to-{output_modality}"
        return task_name in cls.supported_tasks
