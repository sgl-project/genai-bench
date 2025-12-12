"""Customized user for SGLang backends with profiling support."""

import os
import time
from typing import Any, Dict, Optional

import requests

from genai_bench.logging import init_logger
from genai_bench.user.openai_user import OpenAIUser

logger = init_logger(__name__)


class SGLangUser(OpenAIUser):
    """
    SGLang-specific user that extends OpenAIUser.

    This class is used to identify the SGLang backend, while the core
    profiling logic is handled by the SGLangProfiler and orchestrated
    by the CLI.
    """

    BACKEND_NAME = "sglang"


class SGLangProfiler:
    """
    Standalone profiler for SGLang servers.

    Can be used independently of the Locust user for more controlled profiling.
    """

    def __init__(
        self,
        base_url: str,
        output_dir: str = "/tmp/genai_bench_profiles",
        profile_by_stage: bool = False,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.output_dir = output_dir
        self.profile_by_stage = profile_by_stage
        self.api_key = api_key
        self._profile_link = None
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def start_profile(
        self,
        num_steps: int = 5,
        activities: list = None,
        profile_name: str = None,
    ) -> str:
        """
        Start profiling on the SGLang server.

        Args:
            num_steps: Number of forward steps to profile.
            activities: List of activities to profile ("CPU", "GPU", "MEM").
            profile_name: Optional name for this profile run.

        Returns:
            Path to the profile output directory.
        """
        if activities is None:
            activities = ["CPU", "GPU"]

        # Create timestamped output directory
        timestamp = int(time.time())
        if profile_name:
            output_path = os.path.join(self.output_dir, f"{profile_name}_{timestamp}")
        else:
            output_path = os.path.join(self.output_dir, str(timestamp))

        os.makedirs(output_path, exist_ok=True)

        # Save server info
        try:
            response = requests.get(
                f"{self.base_url}/get_server_info",
                headers=self._headers,
                timeout=10,
            )
            if response.status_code == 200:
                import json
                with open(os.path.join(output_path, "server_args.json"), "w") as f:
                    json.dump(response.json(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not fetch server info: {e}")

        # Start profiling
        profile_config = {
            "output_dir": output_path,
            "num_steps": str(num_steps),
            "activities": activities,
            "profile_by_stage": self.profile_by_stage,
            "merge_profiles": False,
        }

        logger.info(f"Starting SGLang profile with config: {profile_config}")

        response = requests.post(
            f"{self.base_url}/start_profile",
            json=profile_config,
            headers=self._headers,
            timeout=600,  # 10 minutes max
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to start profiling: {response.status_code} - {response.text}")

        self._profile_link = output_path
        logger.info(f"Profiling complete. Traces saved to: {output_path}")

        return output_path

    def get_profile_link(self) -> Optional[str]:
        """Get the path to the last profile output."""
        return self._profile_link
