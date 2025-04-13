import os
import sys
import logging
import re
import json
from datetime import datetime

DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("1","true","yes")

# Patterns for sensitive keys
SENSITIVE_PATTERNS = [
    re.compile(r".*TOKEN.*", re.IGNORECASE),
    re.compile(r".*PASSWORD.*", re.IGNORECASE),
    re.compile(r".*KEY.*", re.IGNORECASE),
    re.compile(r".*SECRET.*", re.IGNORECASE),
    re.compile(r".*COOKIE.*", re.IGNORECASE),
]

def mask_if_sensitive(key: str, val: str) -> str:
    for pat in SENSITIVE_PATTERNS:
        if pat.match(key):
            return "******"
    return val

class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Base log object
        obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "dry_run": DRY_RUN,
        }
        # If an exception is present, include its full stack
        if record.exc_info:
            obj["exc_info"] = self.formatException(record.exc_info)
        # Include any extras
        for k, v in record.__dict__.items():
            if k in ("name","msg","args","levelname","levelno","pathname",
                     "filename","module","exc_info","exc_text","stack_info",
                     "lineno","funcName","created","msecs","relativeCreated",
                     "thread","threadName","processName","process"):
                continue
            # Mask if this field might be sensitive
            obj[k] = mask_if_sensitive(k, str(v))
        return json.dumps(obj)

def configure_root_logger():
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    # Remove any existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(logging.INFO)

# Apply immediately
configure_root_logger()

class YTdlpLogger:
    def debug(self, *args, **kwargs):    pass
    def warning(self, *args, **kwargs):  pass
    def error(self, *args, **kwargs):    pass

# Optionally monkey-patch the default so even code that doesn’t pass a logger
# gets silenced. Remove this block if you prefer explicit logger injection.
try:
    import yt_dlp
    yt_dlp.YtdlLogger = YTdlpLogger
except ImportError:
    pass
