"""Custom user implementation that dynamically loads user classes from external modules."""

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Type

from genai_bench.logging import init_logger
from genai_bench.user.base_user import BaseUser

logger = init_logger(__name__)


class CustomUser(BaseUser):
    """Dynamic user class that loads custom backend implementations.

    This class allows users to provide their own backend implementations
    by specifying a Python file path containing a BaseUser subclass.

    Similar to how custom datasets can be loaded, this enables extending
    genai-bench with custom API backends without modifying the core codebase.
    """

    BACKEND_NAME = "custom"

    # These will be set when the custom class is loaded
    supported_tasks = {}
    _custom_class: Optional[Type[BaseUser]] = None
    _custom_module_path: Optional[str] = None

    @classmethod
    def load_custom_class(cls, module_path: str, class_name: Optional[str] = None) -> Type[BaseUser]:
        """Load a custom user class from a Python file.

        Args:
            module_path: Path to the Python file containing the custom user class
            class_name: Name of the class to load. If None, will look for a class
                       that inherits from BaseUser (excluding BaseUser itself)

        Returns:
            The loaded custom user class

        Raises:
            ImportError: If the module cannot be loaded
            AttributeError: If the specified class doesn't exist or no suitable class is found
            TypeError: If the class doesn't inherit from BaseUser
        """
        module_path_obj = Path(module_path)
        if not module_path_obj.exists():
            raise FileNotFoundError(f"Custom backend module not found: {module_path}")

        # Load the module dynamically
        module_name = module_path_obj.stem
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        logger.info(f"Loaded custom backend module from: {module_path}")

        # Get the custom class
        if class_name:
            # Use the specified class name
            if not hasattr(module, class_name):
                raise AttributeError(
                    f"Class '{class_name}' not found in module {module_path}"
                )
            custom_class = getattr(module, class_name)
        else:
            # Auto-detect a BaseUser subclass
            custom_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseUser)
                    and attr is not BaseUser
                    and attr.__module__ == module_name
                ):
                    if custom_class is not None:
                        raise AttributeError(
                            f"Multiple BaseUser subclasses found in {module_path}. "
                            f"Please specify the class name explicitly."
                        )
                    custom_class = attr

            if custom_class is None:
                raise AttributeError(
                    f"No BaseUser subclass found in {module_path}. "
                    f"Please ensure your custom user class inherits from BaseUser."
                )

        # Validate the class
        if not issubclass(custom_class, BaseUser):
            raise TypeError(
                f"Custom class '{custom_class.__name__}' must inherit from BaseUser"
            )

        logger.info(f"Loaded custom user class: {custom_class.__name__}")
        logger.info(f"Supported tasks: {custom_class.supported_tasks}")

        # Store the custom class for later instantiation
        cls._custom_class = custom_class
        cls._custom_module_path = module_path
        cls.supported_tasks = custom_class.supported_tasks

        return custom_class

    def __new__(cls, *args, **kwargs):
        """Create an instance of the custom user class instead of CustomUser."""
        if cls._custom_class is None:
            raise RuntimeError(
                "Custom user class has not been loaded. "
                "Call CustomUser.load_custom_class() first."
            )

        # Create an instance of the actual custom class
        return object.__new__(cls._custom_class)

    def __init__(self, *args, **kwargs):
        """Initialize using the custom class's __init__."""
        if self._custom_class is None:
            raise RuntimeError(
                "Custom user class has not been loaded. "
                "Call CustomUser.load_custom_class() first."
            )

        # The __init__ of the custom class will be called automatically
        # due to the __new__ override
        super().__init__(*args, **kwargs)
