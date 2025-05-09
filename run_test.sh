#!/bin/bash
source "$(dirname "$0")/.venv/bin/activate"
mypy ttnopt && pytest --cov=ttnopt --cov-report=term-missing