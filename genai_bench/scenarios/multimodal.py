import random
from typing import Optional, Tuple

from genai_bench.scenarios.base import MultiModality, Scenario, parse_params_str


class ImageModality(Scenario):
    """
    Image input and text output
    e.g.
    I(256,256) or
    I(2048,2048,2)
    The third number represents the
    number of images
    """

    scenario_type = MultiModality.IMAGE
    validation_pattern = r"^I\(\d+,\d+(?:,\d+)?\)$"

    def __init__(
        self,
        num_input_dimension_width: int,
        num_input_dimension_height: int,
        num_input_images: int = 1,
        max_output_token: Optional[int] = None,
    ):
        self.num_input_dimension_width = num_input_dimension_width
        self.num_input_dimension_height = num_input_dimension_height
        self.num_input_images = num_input_images
        self.max_output_token = max_output_token

    def sample(self) -> Tuple[Tuple[int, int], int, int | None]:
        return (
            (
                self.num_input_dimension_width,
                self.num_input_dimension_height,
            ),
            self.num_input_images,
            self.max_output_token,
        )

    def to_string(self) -> str:  # TODO: include max_output_token in the string
        if self.num_input_images == 1:
            return (
                f"I({self.num_input_dimension_width},{self.num_input_dimension_height})"
            )
        else:
            return (
                f"I({self.num_input_dimension_width},"
                f"{self.num_input_dimension_height},"
                f"{self.num_input_images})"
            )

    @classmethod
    def parse(cls, params_str: str) -> "ImageModality":
        num_input_dimension_width, num_input_dimension_height, *optional = (
            parse_params_str(params_str)[0]
        )
        if not optional:
            return cls(
                num_input_dimension_width=num_input_dimension_width,
                num_input_dimension_height=num_input_dimension_height,
            )
        else:
            return cls(
                num_input_dimension_width=num_input_dimension_width,
                num_input_dimension_height=num_input_dimension_height,
                num_input_images=optional[0],
            )


class AudioScenario(Scenario):
    """
    Audio input scenario with Gaussian-distributed clip duration.

    Format: A(mean_s,std_s)
    e.g. A(10,5) — mean 10s, std 5s (matching Ming's N(10,5) convention)
         A(30,15) — mean 30s, std 15s
         A(60,30) — mean 60s, std 30s

    sample() returns a clip duration in seconds, clamped to [1, mean+3*std].
    """

    scenario_type = MultiModality.AUDIO
    validation_pattern = r"^A\(\d+,\d+\)$"

    def __init__(self, mean_s: int, std_s: int):
        self.mean_s = mean_s
        self.std_s = std_s

    def sample(self) -> float:
        """Sample a clip duration in seconds from N(mean_s, std_s).

        Clamped to [1s, mean+3*std].
        """
        duration = random.gauss(self.mean_s, self.std_s)
        max_duration = self.mean_s + 3 * self.std_s
        return max(1.0, min(duration, float(max_duration)))

    def to_string(self) -> str:
        return f"A({self.mean_s},{self.std_s})"

    @classmethod
    def parse(cls, params_str: str) -> "AudioScenario":
        mean_s, std_s = parse_params_str(params_str)[0]
        return cls(mean_s=mean_s, std_s=std_s)
