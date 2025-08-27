import logging
import sys
from pathlib import Path
from datetime import datetime

# Define project root (adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProjectRelativeFormatter(logging.Formatter):
    """Custom formatter that shows file path relative to project root."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            path = Path(record.pathname).resolve()
            record.relpath = path.relative_to(PROJECT_ROOT)
        except Exception:
            record.relpath = record.pathname  # fallback to absolute
        return super().format(record)


class AlignedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, path_column=100):
        super().__init__(fmt, datefmt)
        self.path_column = path_column

    def format(self, record):
        # Format the original message
        original_message = super().format(record)

        # Split into lines
        lines = original_message.splitlines()

        # Path info to append
        path_info = f"[{record.pathname}:{record.lineno}]"

        # Pad only the last line to align path_info
        line = lines[0]
        padding = max(self.path_column - len(line), 1)
        lines[0] = f"{line}{' ' * padding}{path_info}"

        # Rejoin all lines
        return "\n".join(lines)


# Ensure logs directory exists
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

# Timestamped log file
timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
log_file = log_dir / f"{timestamp}.log"

# Root logger
logger = logging.getLogger("src")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(
    ProjectRelativeFormatter(
        "%(asctime)s | %(name)s | %(levelname)s | [%(relpath)s:%(lineno)d] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
)

# File handler
fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(
    AlignedFormatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        path_column=120,  # column at which [pathname:lineno] should start
    )
)

# --- Add handlers ---
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

logger.info(f"log file: {log_file}")

launch_str = " ".join(sys.orig_argv)
logger.debug(f"running '{launch_str}'")

# Delete old log files
log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime)
max_logs = 10
while len(log_files) >= max_logs:
    oldest = log_files.pop(0)
    try:
        oldest.unlink()
        logger.debug(f"deleted old log file: {oldest}")
    except Exception as e:
        logger.debug(f"failed to delete {oldest}: {e}")

logger.debug("initialized")