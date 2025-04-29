from .api import app
from .config import load_config

load_config()

__all__ = ["app"]
