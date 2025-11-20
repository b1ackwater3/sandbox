#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for SandboxFusion server.
# Usage: HOST=0.0.0.0 PORT=8080 ./run.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Ensure docs site mount exists to avoid startup error
mkdir -p docs/build

# Start server (FastAPI via uvicorn)
exec make run-online HOST="${HOST:-0.0.0.0}" PORT="${PORT:-8080}"

