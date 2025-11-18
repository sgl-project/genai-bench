import logging
import multiprocessing
import os
import sys
from io import StringIO
from typing import List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel


class RollingRichPanelHandler(RichHandler):
    """
    A log handler that emits the record using rich formatting and
    updates the most recent logs to the logs panel on the live dashboard.
    """

    def __init__(
        self, layout: Layout, panel_name="logs", max_entries=10, *args, **kwargs
    ):
        self.log_buffer = StringIO()
        kwargs["console"] = Console(file=self.log_buffer, width=150)
        super().__init__(*args, **kwargs)
        self.layout = layout
        self.panel_name = panel_name
        self.max_entries = max_entries

    def emit(self, record):
        # Continue normal RichHandler behavior
        super().emit(record)
        log_contents = self.log_buffer.getvalue()

        # Trim the log lines to only the last max_entries
        log_lines = log_contents.splitlines()
        last_10_lines = log_lines[-self.max_entries :]
        trimmed_log_contents = "\n".join(last_10_lines)
        self.layout[self.panel_name].update(
            Panel(trimmed_log_contents, title="Logs", border_style="red")
        )


class DelayedRichHandler(RichHandler):
    """
    A delayed RichHandler that buffers all the logs then flush them to console
    all at once later.

    During rich.live.Live context, all the logs that are captured by RichHandler
    will be overridden by the Live dashboard screen. We need a delayed handler
    that doesn't flush the logs until the dashboard exits.
    """

    def __init__(self, live: Live, *args, **kwargs):
        super().__init__(*args, markup=True, **kwargs)
        self.live = live
        self.record_buffer: List[
            logging.LogRecord
        ] = []  # Buffer to hold LogRecord objects temporarily
        self.flush_later = True

    def emit(self, record: logging.LogRecord):
        if self.flush_later:
            # Buffer the LogRecord objects
            self.record_buffer.append(record)
        else:
            # Emit normally after flushing
            super().emit(record)

        # Exit on error logs
        if record.levelno >= logging.ERROR:
            self.flush_buffer()
            sys.exit(1)

    def flush_buffer(self):
        if self.live and self.live.is_started:
            self.live.stop()

        # Flush the buffered LogRecord objects
        self.flush_later = False
        for record in self.record_buffer:
            super().emit(record)
        self.record_buffer.clear()


class WorkerRichHandler(RichHandler):
    """A RichHandler that also forwards logs to master process."""

    def __init__(
        self, worker_id: str, log_queue: multiprocessing.Queue, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.worker_id = worker_id
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        # Forward to master
        self.log_queue.put(
            {
                "worker_id": self.worker_id,
                "message": record.getMessage(),
                "level": record.levelname,
            }
        )


class LoggingManager:
    def __init__(self, command_type, layout=None, live=None, log_dir=None):
        self.command_type = command_type
        self.layout = layout
        self.live = live
        self.log_dir = log_dir
        self.delayed_handler = None
        self.init_logging()

    def init_logging(self):
        """Initialize logging based on command type."""
        log_level = os.getenv("GENAI_BENCH_LOGGING_LEVEL", "INFO").upper()
        enable_ui_str = os.getenv("ENABLE_UI", "true").lower()
        enable_ui = enable_ui_str in ("true", "1", "yes", "on")

        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = "INFO"

        file_handler = self.get_file_handler()

        if self.command_type == "benchmark":
            extra_handlers = (
                self.init_ui_logging() if enable_ui else [self.get_console_handler()]
            )
        else:
            extra_handlers = [self.get_rich_handler()]

        logging.basicConfig(level=log_level, handlers=[file_handler, *extra_handlers])

        # Set up exception handling
        self.setup_exception_handler()

    def setup_exception_handler(self):
        """Set up exception handling to log uncaught exceptions."""

        def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                if self.delayed_handler:
                    self.delayed_handler.flush_buffer()
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            logging.error(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback),
            )

            if self.delayed_handler:
                self.delayed_handler.flush_buffer()

            sys.exit(1)

        sys.excepthook = handle_uncaught_exception

    def get_file_handler(self):
        """Return a file handler for logging."""
        file_log_format = "{levelname:<8} {asctime} - {name}:{funcName} - {message}"
        date_format = "%Y-%m-%d %H:%M:%S.%f"

        # Determine log file path
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_file = os.path.join(self.log_dir, "genai_bench.log")
        else:
            log_file = "genai_bench.log"

        file_handler = logging.FileHandler(log_file)

        file_formatter = logging.Formatter(
            file_log_format, datefmt=date_format, style="{"
        )
        file_handler.setFormatter(file_formatter)

        return file_handler

    @staticmethod
    def get_console_handler():
        """Return a console handler for logging with a standard format."""
        log_format = "{levelname:<8} {asctime} - {name}:{funcName} - {message}"
        date_format = "%Y-%m-%d %H:%M:%S.%f"

        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            log_format, datefmt=date_format, style="{"
        )
        console_handler.setFormatter(console_formatter)
        return console_handler

    @staticmethod
    def get_rich_handler():
        """Return a rich handler for logging."""

        rich_handler = RichHandler(rich_tracebacks=True)
        rich_formatter = logging.Formatter("%(message)s", style="%")
        rich_handler.setFormatter(rich_formatter)
        return rich_handler

    def init_ui_logging(self):
        """Initialize UI logging handlers for 'benchmark'."""
        date_format = "%Y-%m-%d %H:%M:%S.%f"

        # Panel handler to stream logs to the logs panel in dashboard
        panel_handler = RollingRichPanelHandler(
            layout=self.layout,
            rich_tracebacks=True,
            log_time_format=date_format,
        )
        panel_formatter = logging.Formatter(
            "%(message)s", datefmt=date_format, style="%"
        )
        panel_handler.setFormatter(panel_formatter)

        # Delayed handler to flush logs after dashboard exits
        self.delayed_handler = DelayedRichHandler(
            live=self.live,
            console=Console(),
            rich_tracebacks=True,
        )
        delayed_formatter = logging.Formatter("%(message)s", style="%")
        self.delayed_handler.setFormatter(delayed_formatter)

        return [panel_handler, self.delayed_handler]


class WorkerLoggingManager:
    """Manages logging setup for worker processes."""

    def __init__(self, worker_id: str, log_queue: multiprocessing.Queue, log_dir=None):
        self.worker_id = worker_id
        self.log_queue = log_queue
        self.log_dir = log_dir
        self.setup_logging()

    def setup_logging(self):
        """Set up worker-specific logging."""
        log_level = os.getenv("GENAI_BENCH_LOGGING_LEVEL", "INFO").upper()

        # Create worker-specific file handler
        file_handler = self._get_file_handler(
            f"genai_bench_worker_{self.worker_id}.log"
        )

        # Create worker-specific rich handler that forwards to master
        worker_handler = self._create_worker_handler()

        # Configure logging
        logging.basicConfig(
            level=log_level,
            handlers=[file_handler, worker_handler],
            force=True,  # Override any existing configuration
        )

    def _create_worker_handler(self) -> logging.Handler:
        """Create a rich handler that forwards logs to master."""
        handler = WorkerRichHandler(
            worker_id=self.worker_id,
            log_queue=self.log_queue,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", style="%"))
        return handler

    def _get_file_handler(self, filename: str) -> logging.FileHandler:
        """Create a file handler for worker logs."""
        file_log_format = "{levelname:<8} {asctime} - {name}:{funcName} - {message}"
        date_format = "%Y-%m-%d %H:%M:%S.%f"

        # Determine log file path
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_file = os.path.join(self.log_dir, filename)
        else:
            log_file = filename

        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            file_log_format, datefmt=date_format, style="{"
        )
        file_handler.setFormatter(file_formatter)
        return file_handler


def init_logger(name: str):
    return logging.getLogger(name)
