import hashlib
import json
from typing import Any

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from config import config


def get_train_tokenizer(vocab_size: int | None = None):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    kwargs: dict[str, Any] = {
        "special_tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    }
    if vocab_size:
        kwargs["vocab_size"] = vocab_size

    trainer = trainers.BpeTrainer(**kwargs)

    files = [ str(config.SHAKESPEARE_FILEPATH) ]
    tokenizer.train(files, trainer=trainer)

    # save tokenizer to a file
    config_str = json.dumps({
            **kwargs,
            "files": files
        }, sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(config_str).hexdigest()[:8]
    tokenizer_path = config.TOKENIZER_PATH / f"tokenizer_{config_hash}.json"

    tokenizer.save(str(tokenizer_path), pretty=True)

    return tokenizer


def get_tokenizer(vocab_size: int | None = None):
    kwargs: dict[str, Any] = {
        "special_tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    }
    if vocab_size:
        kwargs["vocab_size"] = vocab_size

    files = [ str(config.SHAKESPEARE_FILEPATH) ]
    config_str = json.dumps({
            **kwargs,
            "files": files
        }, sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(config_str).hexdigest()[:8]
    tokenizer_path = config.TOKENIZER_PATH / f"tokenizer_{config_hash}.json"

    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    return get_train_tokenizer(vocab_size)

