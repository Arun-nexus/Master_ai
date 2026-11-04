from functools import lru_cache
from pathlib import Path
from typing import cast

import yaml

from logger import logger
from schemas import Params

BASE_DIR = Path(__file__).resolve().parent
PARAMS_PATH = BASE_DIR / "params.yaml"


@lru_cache(maxsize=1)
def get_params() -> Params:
    """params.yaml ek baar load hota hai aur cache ho jata hai."""
    try:
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        missing = set(Params.__annotations__) - set(data or {})
        if missing:
            raise KeyError(f"params.yaml me ye keys missing hain: {sorted(missing)}")
        logger.info("params file loaded successfully")
        return cast(Params, data)
    except Exception:
        logger.exception("params file was not loaded successfully")
        raise