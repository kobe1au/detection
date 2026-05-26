#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

TARGET="${1:-final}"
"${PYTHON_BIN:-python}" run.py "${TARGET}"
