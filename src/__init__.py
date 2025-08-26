import logging
import sys
from pathlib import Path
from datetime import datetime

# Define project root (adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class ProjectRelativeFormatter(logging.Formatter):
    """Custom formatter that shows file path relative to project root."""

    def format(self, record):
        try:
            path = Path(record.pathname).resolve()
            record.relpath = path.relative_to(PROJECT_ROOT)
        except Exception:
            record.relpath = record.pathname  # fallback to absolute
        return super().format(record)

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
    ProjectRelativeFormatter(
        # abs path so vscode sees it as a URI in the log file
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"+"\t"*5+"[%(pathname)s:%(lineno)d]",
        "%Y-%m-%d %H:%M:%S",
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

logger.debug("initialized")