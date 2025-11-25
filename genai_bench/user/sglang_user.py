"""Customized user for SGLang backends with profiling support."""

import os
import time
from typing import Any, Dict, Optional

import requests
from locust import task

from genai_bench.logging import init_logger
from genai_bench.user.openai_user import OpenAIUser

logger = init_logger(__name__)


class SGLangUser(OpenAIUser):
    """
    SGLang-specific user that extends OpenAIUser with profiling capabilities.

    SGLang servers expose a /start_profile endpoint that captures Perfetto traces
    for GPU kernel-level performance analysis. This user class integrates that
    profiling capability into genai-bench.
    """

    BACKEND_NAME = "sglang"

    # Class-level flag to ensure only one worker starts profiling
    _profile_started = False

    def on_start(self):
        """Initialize the user and optionally start profiling."""
        super().on_start()

        # Check if profiling is enabled via environment options
        if not hasattr(self.environment, 'parsed_options'):
            return

        opts = self.environment.parsed_options
        if not getattr(opts, 'sglang_profile', False):
            return

        # Only the first worker should start profiling
        if SGLangUser._profile_started:
            logger.debug("Profiling already started by another worker, skipping.")
            return

        SGLangUser._profile_started = True
        self._start_profiling(opts)

    def _start_profiling(self, opts):
        """
        Start SGLang server-side profiling.

        Args:
            opts: Parsed command line options containing profile settings.
        """
        profile_output_dir = getattr(opts, 'sglang_profile_output_dir', '/tmp/genai_bench_profiles')
        profile_steps = getattr(opts, 'sglang_profile_steps', 10)
        profile_by_stage = getattr(opts, 'sglang_profile_by_stage', True)

        # Create output directory
        os.makedirs(profile_output_dir, exist_ok=True)

        # Build profile request
        profile_config = {
            "output_dir": profile_output_dir,
            "num_steps": str(profile_steps),
            "activities": ["CPU", "GPU"],
            "profile_by_stage": profile_by_stage,
            "merge_profiles": False,
        }

        logger.info(f"Starting SGLang profiling: {profile_config}")

        try:
            # First, get server info for metadata
            server_info_response = requests.get(
                f"{self.host}/get_server_info",
                timeout=10
            )
            if server_info_response.status_code == 200:
                server_info = server_info_response.json()
                logger.info(f"SGLang server info: model={server_info.get('model_path', 'unknown')}")

            # Start profiling (this is async - server profiles next N steps)
            # Note: We don't wait for completion here as profiling happens
            # during the benchmark run
            response = requests.post(
                f"{self.host}/start_profile",
                json=profile_config,
                timeout=300  # Profiling can take a while
            )

            if response.status_code == 200:
                logger.info(f"SGLang profiling started successfully. Traces will be saved to: {profile_output_dir}")
            else:
                logger.warning(f"Failed to start profiling: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error starting SGLang profiling: {e}")

    @classmethod
    def reset_profile_state(cls):
        """Reset the profile started flag. Useful for testing."""
        cls._profile_started = False


class SGLangProfiler:
    """
    Standalone profiler for SGLang servers.

    Can be used independently of the Locust user for more controlled profiling.
    """

    def __init__(
        self,
        base_url: str,
        output_dir: str = "/tmp/genai_bench_profiles",
        profile_by_stage: bool = True,
    ):
        self.base_url = base_url.rstrip('/')
        self.output_dir = output_dir
        self.profile_by_stage = profile_by_stage
        self._profile_link = None

    def start_profile(
        self,
        num_steps: int = 10,
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
            response = requests.get(f"{self.base_url}/get_server_info", timeout=10)
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
            timeout=600  # 10 minutes max
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to start profiling: {response.status_code} - {response.text}")

        self._profile_link = output_path
        logger.info(f"Profiling complete. Traces saved to: {output_path}")

        return output_path

    def get_profile_link(self) -> Optional[str]:
        """Get the path to the last profile output."""
        return self._profile_link
