"""Run logging -- what the pipeline is doing, while it is doing it.

Two rules shape this module:

  1. Logs go to **stderr**. `run_fraud.py single` writes its JSON verdict to
     stdout, so `run_fraud.py single --id X | jq` must stay clean no matter how
     chatty the run is.
  2. Nothing is emitted unless a CLI calls `configure()`. Importing `detector`
     from a notebook or from `tests.py` produces silence, not log spam.

Three verbosities, chosen so the default answers "what is happening right now"
without burying it:

  -q        WARNING  only things that went wrong
  (default) INFO     stage transitions, progress, counts
  -v        DEBUG    per-record scores, every SLM call and its latency
"""

import logging
import sys
import time

ROOT = "fraud"
_STARTED = time.time()

# Importing a module must never print. A NullHandler on the package root keeps
# logging's "no handlers found" fallback quiet until configure() runs.
logging.getLogger(ROOT).addHandler(logging.NullHandler())

_LABELS = {
    "DEBUG": "debug",
    "INFO": "info ",
    "WARNING": "WARN ",
    "ERROR": "ERROR",
    "CRITICAL": "FATAL",
}


class _ElapsedFormatter(logging.Formatter):
    """`[  12.4s] info  batch    scored 300/988` -- elapsed time, level, stage."""

    def format(self, record):
        elapsed = time.time() - _STARTED
        stage = record.name.split(".")[-1]
        label = _LABELS.get(record.levelname, record.levelname)
        text = f"  [{elapsed:6.1f}s] {label}  {stage:<9} {record.getMessage()}"
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


def configure(verbose: int = 0, quiet: bool = False, stream=None) -> None:
    """Attach the one stderr handler. Called by CLIs only, never on import."""
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    root = logging.getLogger(ROOT)
    root.setLevel(level)
    for handler in list(root.handlers):
        if not isinstance(handler, logging.NullHandler):
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(_ElapsedFormatter())
    root.addHandler(handler)
    root.propagate = False


def get(name: str) -> logging.Logger:
    """`log.get(__name__)` -> a child of the package root."""
    return logging.getLogger(f"{ROOT}.{name}")


def elapsed() -> float:
    return time.time() - _STARTED


class Progress:
    """Periodic progress with a rate and an ETA.

    A full batch makes ~150 SLM calls at a second or two each, so a bare
    "scored 300/988" leaves the user guessing how long the rest will take.
    """

    def __init__(self, logger, total: int, every: int = 100, unit: str = "records"):
        self.logger = logger
        self.total = total
        self.every = max(1, every)
        self.unit = unit
        self.started = time.time()

    def update(self, done: int, suffix: str = "") -> None:
        if done % self.every and done != self.total:
            return
        spent = max(1e-6, time.time() - self.started)
        rate = done / spent
        remaining = (self.total - done) / rate if rate else 0.0
        eta = f", eta {remaining:.0f}s" if done < self.total else ""
        self.logger.info(f"scored {done}/{self.total} {self.unit} "
                         f"({rate:.1f}/s{eta}){suffix}")

    def done(self) -> float:
        return time.time() - self.started
