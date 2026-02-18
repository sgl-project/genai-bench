"""Tests for multimodal scenario implementations."""

from genai_bench.scenarios.multimodal import ImageModality, VideoModality


def test_image_modality_creation():
    """Test ImageModality creation."""
    scenario = ImageModality(
        num_input_dimension_width=256,
        num_input_dimension_height=256,
        num_input_images=1,
        max_output_token=100,
    )

    assert scenario.num_input_dimension_width == 256
    assert scenario.num_input_dimension_height == 256
    assert scenario.num_input_images == 1
    assert scenario.max_output_token == 100


def test_image_modality_sampling():
    """Test ImageModality sampling."""
    scenario = ImageModality(
        num_input_dimension_width=512,
        num_input_dimension_height=512,
        num_input_images=2,
        max_output_token=200,
    )

    dimensions, num_images, max_tokens = scenario.sample()
    assert dimensions == (512, 512)
    assert num_images == 2
    assert max_tokens == 200


def test_image_modality_default_values():
    """Test ImageModality with default values."""
    scenario = ImageModality(
        num_input_dimension_width=256, num_input_dimension_height=256
    )

    assert scenario.num_input_images == 1
    assert scenario.max_output_token is None


def test_image_modality_to_string():
    """Test ImageModality string representation."""
    # With default num_input_images
    scenario = ImageModality(
        num_input_dimension_width=256, num_input_dimension_height=256
    )
    assert scenario.to_string() == "I(256,256)"

    # With multiple images
    scenario = ImageModality(
        num_input_dimension_width=512,
        num_input_dimension_height=512,
        num_input_images=3,
    )
    assert scenario.to_string() == "I(512,512,3)"


def test_image_modality_parse():
    """Test ImageModality parsing from string."""
    # Parse simple format
    scenario = ImageModality.parse("(256,256)")
    assert scenario.num_input_dimension_width == 256
    assert scenario.num_input_dimension_height == 256
    assert scenario.num_input_images == 1

    # Parse with multiple images
    scenario = ImageModality.parse("(1024,1024,2)")
    assert scenario.num_input_dimension_width == 1024
    assert scenario.num_input_dimension_height == 1024
    assert scenario.num_input_images == 2


def test_image_modality_from_string():
    """Test ImageModality creation from string."""
    from genai_bench.scenarios.base import Scenario

    scenario = Scenario.from_string("I(256,256)")
    assert isinstance(scenario, ImageModality)
    assert scenario.num_input_dimension_width == 256
    assert scenario.num_input_dimension_height == 256

    scenario = Scenario.from_string("I(512,512,3)")
    assert isinstance(scenario, ImageModality)
    assert scenario.num_input_images == 3


def test_video_modality_creation():
    """Test VideoModality creation."""
    scenario = VideoModality(
        max_output_token=100,
    )

    assert scenario.max_output_token == 100
    assert scenario.num_input_videos == 1  # default


def test_video_modality_sampling():
    """Test VideoModality sampling."""
    scenario = VideoModality(
        num_input_videos=2,
        max_output_token=256,
    )

    num_videos, max_tokens = scenario.sample()
    assert num_videos == 2
    assert max_tokens == 256


def test_video_modality_to_string():
    """Test VideoModality string representation."""
    # Single video (default)
    scenario = VideoModality()
    assert scenario.to_string() == "V(1)"

    # Multiple videos
    scenario_multi = VideoModality(
        num_input_videos=3,
    )
    assert scenario_multi.to_string() == "V(3)"


def test_video_modality_parse():
    """Test VideoModality parsing from string."""
    # Single video
    scenario = VideoModality.parse("(1)")
    assert scenario.num_input_videos == 1

    # Multiple videos
    scenario_multi = VideoModality.parse("(2)")
    assert scenario_multi.num_input_videos == 2


def test_video_modality_from_string():
    """Test VideoModality creation from string."""
    from genai_bench.scenarios.base import Scenario

    scenario = Scenario.from_string("V(1)")
    assert isinstance(scenario, VideoModality)
    assert scenario.num_input_videos == 1

    scenario_multi = Scenario.from_string("V(2)")
    assert isinstance(scenario_multi, VideoModality)
    assert scenario_multi.num_input_videos == 2
