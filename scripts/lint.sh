#!/usr/bin/env bash

set -e
set -x

mypy sieval
ruff check sieval tests scripts
ruff format sieval tests scripts --check
