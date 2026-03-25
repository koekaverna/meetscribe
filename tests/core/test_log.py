"""Tests for logging module: StructuredFormatter and apply_log_level."""

import logging

from meetscribe.log import StructuredFormatter, apply_log_level


class TestApplyLogLevel:
    def test_sets_root_logger_level(self) -> None:
        apply_log_level("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_invalid_level_falls_back_to_info(self) -> None:
        apply_log_level("NONSENSE")
        assert logging.getLogger().level == logging.INFO

    def test_sets_file_handler_level(self, tmp_path) -> None:
        root = logging.getLogger()
        handler = logging.FileHandler(tmp_path / "test.log")
        root.addHandler(handler)
        try:
            apply_log_level("WARNING")
            assert handler.level == logging.WARNING
        finally:
            root.removeHandler(handler)
            handler.close()


class TestStructuredFormatter:
    def _make_record(self, msg: str, **extras) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for k, v in extras.items():
            setattr(record, k, v)
        return record

    def test_no_extras(self) -> None:
        fmt = StructuredFormatter("%(message)s")
        record = self._make_record("hello")
        assert fmt.format(record) == "hello"

    def test_with_extras(self) -> None:
        fmt = StructuredFormatter("%(message)s")
        record = self._make_record("hello", foo=42, bar="baz")
        result = fmt.format(record)
        assert result.startswith("hello | ")
        assert "foo=42" in result
        assert 'bar="baz"' in result

    def test_string_extras_are_json_quoted(self) -> None:
        fmt = StructuredFormatter("%(message)s")
        record = self._make_record("msg", path="/a/b")
        result = fmt.format(record)
        assert '"/a/b"' in result
