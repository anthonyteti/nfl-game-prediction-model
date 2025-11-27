"""Project-wide configuration constants."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Seasons inclusive start/end
SEASONS = list(range(2010, 2024))


def ensure_directories() -> None:
    """Create key directories if they do not already exist."""
    for directory in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "MODELS_DIR",
    "SEASONS",
    "ensure_directories",
]
