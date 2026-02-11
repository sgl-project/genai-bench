"""User class for OCI's OpenAI-compatible endpoints."""

import time
from typing import Optional

import httpx
from locust import task
from oci_openai import (
    OciInstancePrincipalAuth,
    OciResourcePrincipalAuth,
    OciSessionAuth,
    OciUserPrincipalAuth,
)
from openai import OpenAI

from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserImageGenerationRequest,
    UserImageGenerationResponse,
)
from genai_bench.user.openai_user import OpenAIUser

logger = init_logger(__name__)

OCI_AUTH_CLASS_MAP = {
    "oci_security_token": OciSessionAuth,
    "oci_user_principal": OciUserPrincipalAuth,
    "oci_instance_principal": OciInstancePrincipalAuth,
    "oci_obo_token": OciResourcePrincipalAuth,
}


class OCIOpenAIUser(OpenAIUser):
    """User class for OCI's OpenAI-compatible endpoints.

    Overrides authentication (OCI request signing) and image generation.
    """

    oci_profile: Optional[str] = None
    oci_config_file: Optional[str] = None
    oci_compartment_id: Optional[str] = None

    def on_start(self):
        if not self.host or not self.auth_provider:
            raise ValueError("Host and auth_provider must be set for OCIOpenAIUser.")
        auth_type = self.auth_provider.get_auth_type()
        auth_cls = OCI_AUTH_CLASS_MAP.get(auth_type)
        if auth_cls is None:
            raise ValueError(
                f"Unsupported OCI auth type: {auth_type}. "
                f"Supported: {list(OCI_AUTH_CLASS_MAP)}"
            )

        if auth_type in ("oci_security_token", "oci_user_principal"):
            kwargs = {"profile_name": self.oci_profile or "DEFAULT"}
            if self.oci_config_file:
                kwargs["config_file"] = self.oci_config_file
            oci_auth = auth_cls(**kwargs)
        else:
            oci_auth = auth_cls()

        self.openai_client = OpenAI(
            api_key="OCI",
            base_url=self.host,
            http_client=httpx.Client(
                auth=oci_auth,
                headers={"CompartmentId": self.oci_compartment_id},
            ),
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

        start_time = time.monotonic()
        try:
            sdk_params = {
                k: v
                for k, v in user_request.additional_request_params.items()
                if k != "compartmentId"
            }

            response = self.openai_client.images.generate(
                model=user_request.model,
                prompt=user_request.prompt,
                n=user_request.num_images,
                size=user_request.size,
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
                tokens_received=len(generated_images),
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
                tokens_received=0,
            )

        self.collect_metrics(metrics_response, "/v1/images/generations")
        return metrics_response
