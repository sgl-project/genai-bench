"""User class for OCI's OpenAI-compatible endpoints."""

from locust import task

import time

import httpx
from oci_openai import (
    OciInstancePrincipalAuth,
    OciResourcePrincipalAuth,
    OciSessionAuth,
    OciUserPrincipalAuth,
)
from openai import OpenAI

from genai_bench.logging import init_logger
from genai_bench.protocol import UserImageGenerationRequest, UserImageGenerationResponse
from genai_bench.user.openai_user import OpenAIUser

logger = init_logger(__name__)

OCI_AUTH_CLASS_MAP = {
    "oci_security_token": OciSessionAuth,
    "oci_user_principal": OciUserPrincipalAuth,
    "oci_instance_principal": OciInstancePrincipalAuth,
    "oci_obo_token": OciResourcePrincipalAuth,
}


class OCIOpenAIUser(OpenAIUser):
    """User class for OCI's OpenAI-compatible endpoints."""

    BACKEND_NAME = "oci-openai"
    supported_tasks = {
        "text-to-image": "image_generation",
    }

    def on_start(self):
        if not self.host or not self.auth_provider:
            raise ValueError(
                "Host and Auth is required for OCIOpenAIUser."
            )  # what's host, what is self.host?
        auth_type = self.auth_provider.get_auth_type()
        auth_cls = OCI_AUTH_CLASS_MAP.get(auth_type)
        if auth_cls is None:
            raise ValueError(
                f"Unsupported OCI auth type: {auth_type}. "
                f"Supported: {list(OCI_AUTH_CLASS_MAP)}"
            )

        if auth_type in ("oci_security_token", "oci_user_principal"):
            profile = getattr(self.auth_provider.oci_auth, "profile", "DEFAULT")
            config_file = getattr(self.auth_provider.oci_auth, "config_path", None)
            kwargs = {"profile_name": profile}
            if config_file:
                kwargs["config_file"] = config_file
            oci_auth = auth_cls(**kwargs)
        else:
            oci_auth = auth_cls()

        self.openai_client = OpenAI(
            api_key="OCI",
            base_url=self.host,
            http_client=httpx.Client(auth=oci_auth),
        )
        self.api_backend = getattr(self, "api_backend", self.BACKEND_NAME)
        super(OpenAIUser, self).on_start()

    @task
    def image_generation(self):
        user_request = self.sample()

        if not isinstance(user_request, UserImageGenerationRequest):
            raise AttributeError(
                f"user_request should be of type "
                f"UserImageGenerationRequest for OCIOpenAIUser."
                f"image_generation, got {type(user_request)}"
            )

        compartment_id = user_request.additional_request_params.get("compartmentId")
        if not compartment_id:
            raise ValueError("compartmentId missing in additional request params")

        start_time = time.monotonic()
        try:
            # Filter out keys already passed as explicit SDK args
            sdk_params = {
                k: v
                for k, v in user_request.additional_request_params.items()
                if k not in ("compartmentId", "model", "prompt", "n", "size")
            }

            response = self.openai_client.images.generate(
                model=user_request.model,
                prompt=user_request.prompt,
                n=user_request.num_images,
                size=user_request.size,
                extra_headers={"CompartmentId": compartment_id},
                **sdk_params,
            )

            end_time = time.monotonic()
            generated_images = [img.url or img.b64_json for img in response.data if img]
            revised_prompt = response.data[0].revised_prompt if response.data else None

            metrics_response = UserImageGenerationResponse(
                status_code=200,
                start_time=start_time,
                end_time=end_time,
                time_at_first_token=end_time,
                generated_images=generated_images,
                revised_prompt=revised_prompt,
                num_prefill_tokens=0,
                images_generated=len(generated_images),
            )

        except Exception as e:
            logger.error(f"OCI image generation failed: {e}")
            metrics_response = UserImageGenerationResponse(
                status_code=getattr(e, "status_code", 500),
                error_message=str(e),
                start_time=start_time,
                end_time=time.monotonic(),
                time_at_first_token=None,
                generated_images=[],
                revised_prompt=None,
                num_prefill_tokens=0,
                images_generated=0,
            )

        self.collect_metrics(metrics_response, "/v1/images/generations")
        return metrics_response
