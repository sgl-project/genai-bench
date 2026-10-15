import base64
from io import BytesIO

import pytest
from PIL import Image

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.protocol import UserImageChatRequest, UserImageEmbeddingRequest
from genai_bench.sampling.image import ImageSampler


def make_image_data_url(image_format, mode="RGB"):
    with BytesIO() as stream:
        Image.new(mode, (2, 3)).save(stream, format=image_format)
        payload = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{payload}"


@pytest.mark.parametrize(
    "image_format,mode", [("PNG", "RGBA"), ("PNG", "L"), ("JPEG", "RGB")]
)
@pytest.mark.parametrize("resize", [None, (4, 5)])
def test_process_image_data_url(image_format, mode, resize):
    data_url = make_image_data_url(image_format, mode)

    result = ImageSampler.process_image(data_url, resize=resize)

    assert result.startswith("data:image/jpeg;base64,")
    with Image.open(BytesIO(base64.b64decode(result.split(",", 1)[1]))) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.mode == "RGB"
        assert decoded.size == (resize or (2, 3))


def test_process_image_accepts_its_own_output():
    data_url = ImageSampler.process_image(Image.new("RGB", (2, 3)))

    result = ImageSampler.process_image(data_url)

    with Image.open(BytesIO(base64.b64decode(result.split(",", 1)[1]))) as decoded:
        assert decoded.size == (2, 3)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
@pytest.mark.parametrize("row_format", ["tuple", "dict"])
@pytest.mark.parametrize(
    "output_modality,request_type",
    [("text", UserImageChatRequest), ("embeddings", UserImageEmbeddingRequest)],
)
def test_sample_image_data_url(image_format, row_format, output_modality, request_type):
    data_url = make_image_data_url(image_format)
    config = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="local-fixture"),
        prompt_column="prompt",
        image_column="image",
    )
    row = (
        ("Describe this image", data_url)
        if row_format == "tuple"
        else {"prompt": "Describe this image", "image": data_url}
    )
    sampler = ImageSampler(
        tokenizer=None,
        model="local-model",
        output_modality=output_modality,
        data=[row],
        dataset_config=config,
    )

    request = sampler.sample(scenario=None)

    assert isinstance(request, request_type)
    assert request.num_images == 1
    assert len(request.image_content) == 1
    assert request.image_content[0].startswith("data:image/jpeg;base64,")
    with Image.open(
        BytesIO(base64.b64decode(request.image_content[0].split(",", 1)[1]))
    ) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.size == (2, 3)


@pytest.mark.parametrize(
    "data_url",
    [
        "data:image/png;base64",
        "data:image/png,not-base64",
        "data:image/png;base64,",
        "data:image/png;base64,%%%",
        "data:image/png;base64,YQ==",
        "data:image/png;base64,a",
        "data:image/png;base64,非ASCII",
    ],
)
def test_process_image_rejects_invalid_data_url(data_url):
    with pytest.raises(ValueError):
        ImageSampler.process_image(data_url)
