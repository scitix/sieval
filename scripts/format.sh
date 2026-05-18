#!/usr/bin/env bash
set -e
set -x

ruff check sieval tests scripts --fix
ruff format sieval tests scripts
