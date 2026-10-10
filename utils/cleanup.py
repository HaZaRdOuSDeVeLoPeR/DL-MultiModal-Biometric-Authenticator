import shutil
from config import PROJECT_ROOT

CACHE_FOLDER = "__pycache__"
root_cache = PROJECT_ROOT / CACHE_FOLDER
src_cache = PROJECT_ROOT / "src" / CACHE_FOLDER
utils_cache = PROJECT_ROOT / "utils" / CACHE_FOLDER
experiments_cache = PROJECT_ROOT / "experiments" / CACHE_FOLDER

CACHES = [root_cache, src_cache, utils_cache, experiments_cache]

def cache_cleanup() -> None:
    caches = [cache for cache in CACHES if cache.is_dir()]
    for cache in caches:
        shutil.rmtree(cache)