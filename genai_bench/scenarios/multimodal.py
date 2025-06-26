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
