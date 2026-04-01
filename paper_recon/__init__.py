import os
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".local.env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
