#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-final}"
exec bash scripts/run_train_2026.sh "${TARGET}"
