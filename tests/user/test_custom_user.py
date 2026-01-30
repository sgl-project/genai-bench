import pytest

from genai_bench.user.custom_user import CustomUser


def test_load_custom_class_valid_file(tmp_path):
    """Test loading a valid custom backend file."""
    content = '''
from genai_bench.user.base_user import BaseUser
class TestUser(BaseUser):
    BACKEND_NAME = "test"
    supported_tasks = {"text-to-text": "chat"}
    def on_start(self): pass
'''
    backend_file = tmp_path / "valid_backend.py"
    backend_file.write_text(content)

    result = CustomUser.load_custom_class(str(backend_file))
    assert result.BACKEND_NAME == "test"
    assert CustomUser.supported_tasks == {"text-to-text": "chat"}


def test_load_custom_class_with_explicit_class_name(tmp_path):
    """Test loading with explicit class name."""
    content = '''
from genai_bench.user.base_user import BaseUser
class MyCustomUser(BaseUser):
    BACKEND_NAME = "my-custom"
    supported_tasks = {"text-to-text": "chat"}
    def on_start(self): pass
'''
    backend_file = tmp_path / "named_backend.py"
    backend_file.write_text(content)

    result = CustomUser.load_custom_class(str(backend_file), class_name="MyCustomUser")
    assert result.BACKEND_NAME == "my-custom"


def test_load_custom_class_file_not_found():
    """Test FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        CustomUser.load_custom_class("/nonexistent/path.py")


def test_load_custom_class_no_baseuser_subclass(tmp_path):
    """Test AttributeError when no BaseUser subclass exists."""
    content = "class NotAUser: pass"
    backend_file = tmp_path / "no_baseuser.py"
    backend_file.write_text(content)

    with pytest.raises(AttributeError, match="No BaseUser subclass found"):
        CustomUser.load_custom_class(str(backend_file))


def test_load_custom_class_multiple_subclasses_no_name(tmp_path):
    """Test AttributeError when multiple BaseUser subclasses without explicit name."""
    content = '''
from genai_bench.user.base_user import BaseUser
class UserOne(BaseUser):
    BACKEND_NAME = "one"
    supported_tasks = {}
    def on_start(self): pass
class UserTwo(BaseUser):
    BACKEND_NAME = "two"
    supported_tasks = {}
    def on_start(self): pass
'''
    backend_file = tmp_path / "multiple_classes.py"
    backend_file.write_text(content)

    with pytest.raises(AttributeError, match="Multiple BaseUser subclasses found"):
        CustomUser.load_custom_class(str(backend_file))


def test_load_custom_class_explicit_class_not_found(tmp_path):
    """Test AttributeError when specified class doesn't exist."""
    content = '''
from genai_bench.user.base_user import BaseUser
class ExistingUser(BaseUser):
    BACKEND_NAME = "existing"
    supported_tasks = {}
    def on_start(self): pass
'''
    backend_file = tmp_path / "existing_backend.py"
    backend_file.write_text(content)

    with pytest.raises(AttributeError, match="not found in module"):
        CustomUser.load_custom_class(str(backend_file), class_name="NonExistentUser")


def test_custom_user_new_without_loading():
    """Test RuntimeError when instantiating without loading first."""
    CustomUser._custom_class = None
    with pytest.raises(RuntimeError, match="not been loaded"):
        CustomUser()
