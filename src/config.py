from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

    TOKENIZER_PATH: Path = ARTIFACTS_DIR

    SHAKESPEARE_FILEPATH: Path = DATA_DIR / "tiny-shakespeare.txt"

config = Config()
