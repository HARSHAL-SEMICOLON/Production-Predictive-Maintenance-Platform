# Sentinel AI - Local Launch PowerShell Script
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Sentinel AI - Production Predictive Maintenance Platform" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if model artifacts exist
if (-not (Test-Path "models/artifacts/xgboost_model.joblib")) {
    Write-Host "[1/3] Pre-trained models not found. Running training pipeline..." -ForegroundColor Yellow
    py -3.12 -m src.models.train --dataset FD001 --epochs 10 --quick
} else {
    Write-Host "[1/3] Pre-trained models found in models/artifacts/." -ForegroundColor Green
}

Write-Host "[2/3] Launching FastAPI Backend (http://127.0.0.1:8000)..." -ForegroundColor Green
$apiProcess = Start-Process -FilePath "py" -ArgumentList "-3.12 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload" -PassThru

Start-Sleep -Seconds 3

Write-Host "[3/3] Launching Streamlit Operations Dashboard (http://localhost:8501)..." -ForegroundColor Green
$dashProcess = Start-Process -FilePath "py" -ArgumentList "-3.12 -m streamlit run dashboard/app.py --server.port 8501" -PassThru

Write-Host "`nPlatform running successfully!" -ForegroundColor Cyan
Write-Host "FastAPI Swagger Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "Streamlit Dashboard:  http://localhost:8501" -ForegroundColor White
Write-Host "Prometheus Metrics:   http://127.0.0.1:8000/metrics" -ForegroundColor White
Write-Host "`nPress Ctrl+C or close console windows to terminate." -ForegroundColor Yellow
