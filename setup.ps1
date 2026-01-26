# Setup script for bank-ml-decisioning project (Windows PowerShell)
# Run this once to prepare the project for demo

Write-Host "=== Bank ML Decisioning - Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..."
$pythonVersion = python --version 2>&1
Write-Host "Python version: $pythonVersion"

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& .venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
Write-Host "Creating directories..."
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data\requests" | Out-Null
New-Item -ItemType Directory -Force -Path "data\mlflow_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "data\rag_index" | Out-Null
New-Item -ItemType Directory -Force -Path "mlflow\artifacts" | Out-Null
New-Item -ItemType Directory -Force -Path "reports\figures" | Out-Null

# Download dataset if not exists
if (-not (Test-Path "data\raw\credit-g.csv")) {
    Write-Host "Downloading dataset (this may take a moment)..."
    python -c "from src.data import load_credit_g; load_credit_g()"
    Write-Host "Dataset downloaded to data\raw\credit-g.csv"
} else {
    Write-Host "Dataset already exists: data\raw\credit-g.csv"
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Start MLflow server:"
Write-Host "   mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts"
Write-Host ""
Write-Host "2. Train a model (in another terminal):"
Write-Host "   .venv\Scripts\Activate.ps1"
Write-Host "   python -m src.train --experiment bank-credit-risk-http --run-name rf-baseline --model rf"
Write-Host ""
Write-Host "3. Select best model:"
Write-Host "   python src/scripts/select_best_model.py"
Write-Host ""
Write-Host "4. Start API:"
Write-Host "   uvicorn src.api.app:app --reload --port 8000"
Write-Host ""
Write-Host "5. Start Streamlit UI:"
Write-Host "   streamlit run streamlit_app.py"
Write-Host ""
