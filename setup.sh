#!/bin/bash
# Setup script for bank-ml-decisioning project
# Run this once to prepare the project for demo

set -e

echo "=== Bank ML Decisioning - Setup ==="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/requests
mkdir -p data/mlflow_cache
mkdir -p data/rag_index
mkdir -p mlflow/artifacts
mkdir -p reports/figures

# Download dataset if not exists
if [ ! -f "data/raw/credit-g.csv" ]; then
    echo "Downloading dataset (this may take a moment)..."
    python3 -c "from src.data import load_credit_g; load_credit_g()"
    echo "Dataset downloaded to data/raw/credit-g.csv"
else
    echo "Dataset already exists: data/raw/credit-g.csv"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Start MLflow server:"
echo "   mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts"
echo ""
echo "2. Train a model (in another terminal):"
echo "   source .venv/bin/activate"
echo "   python -m src.train --experiment bank-credit-risk-http --run-name rf-baseline --model rf"
echo ""
echo "3. Select best model:"
echo "   python src/scripts/select_best_model.py"
echo ""
echo "4. Start API:"
echo "   uvicorn src.api.app:app --reload --port 8000"
echo ""
echo "5. Start Streamlit UI:"
echo "   streamlit run streamlit_app.py"
echo ""
