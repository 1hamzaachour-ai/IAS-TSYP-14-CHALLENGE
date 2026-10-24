#!/usr/bin/env bash
# One-click setup + launch (macOS / Linux): venv -> dependencies -> data -> model -> dashboard
set -euo pipefail
cd "$(dirname "$0")"

[ -x .venv/bin/python ] || python3 -m venv .venv
PY=.venv/bin/python

"$PY" -m pip install --upgrade pip --quiet
"$PY" -m pip install -r requirements.txt --quiet
"$PY" dataset_builder.py
"$PY" train_model.py
exec "$PY" -m streamlit run app.py
