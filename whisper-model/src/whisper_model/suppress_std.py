import contextlib
import io
import logging
import sys


class SuppressStd(contextlib.ContextDecorator):
    def __init__(self, logger: logging.Logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._stdout = None
        self._stderr = None
        self._stringio = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._stringio = io.StringIO()
        sys.stdout = self._stringio
        sys.stderr = self._stringio
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        output = self._stringio.getvalue() if self._stringio else None
        if output:
            self.logger.log(self.level, output.strip())
        if self._stringio:
            self._stringio.close()
        return False
