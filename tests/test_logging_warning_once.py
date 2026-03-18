import logging

from genai_bench.logging import init_logger, warning_once


def test_warning_once_logs_only_first_time(caplog):
    logger = init_logger(__name__)

    with caplog.at_level(logging.WARNING):
        warning_once(logger, "test_key", "first")
        warning_once(logger, "test_key", "second")

    # Only the first message should be logged for this key
    assert any("first" in record.getMessage() for record in caplog.records)
    assert not any("second" in record.getMessage() for record in caplog.records)


def test_warning_once_separate_keys_and_loggers(caplog):
    logger_a = init_logger("logger_a")
    logger_b = init_logger("logger_b")

    with caplog.at_level(logging.WARNING):
        warning_once(logger_a, "key", "message_a_key")
        warning_once(logger_b, "key", "message_b_key")
        warning_once(logger_a, "other_key", "message_a_other")

    messages = [record.getMessage() for record in caplog.records]
    assert "message_a_key" in messages
    assert "message_b_key" in messages
    assert "message_a_other" in messages
