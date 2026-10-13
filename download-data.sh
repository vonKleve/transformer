#!/bin/bash
set -e

mkdir -p data

curl -o ./data/tiny-shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
