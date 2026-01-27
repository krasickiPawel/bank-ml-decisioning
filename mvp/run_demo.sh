#!/bin/bash
# Uruchom MVP demo (train + info jak startować API i Streamlit).
# Użycie: cd mvp && ./run_demo.sh
set -e
cd "$(dirname "$0")"
echo "=== MVP Credit Risk — train ==="
python train.py
echo ""
echo "=== Gotowe. Teraz uruchom w dwóch osobnych terminalach: ==="
echo "  Terminal 1:  uvicorn api:app --host 0.0.0.0 --port 8000"
echo "  Terminal 2:  streamlit run streamlit_app.py"
echo ""
echo "Potem otwórz http://localhost:8501 i ustaw API_URL = http://localhost:8000"
