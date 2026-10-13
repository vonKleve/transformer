from tokenizers import Tokenizer
from torch import Tensor

from tokenizer import get_tokenizer, get_batch
from config import config

tokenizer = get_tokenizer(vocab_size=2000)

encoded_txt = tokenizer.encode("Hello there, general Kenobi")
print(encoded_txt.tokens)
print(encoded_txt.ids)
print(tokenizer.decode(encoded_txt.ids))

inputs, outputs = get_batch(Tensor(encoded_txt.ids), 4, 2)
print(inputs, outputs)
