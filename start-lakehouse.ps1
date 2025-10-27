# Start the transcription lakehouse environment with UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING="utf-8"

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
    Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Virtual environment not found. Run 'python -m venv venv' first." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] UTF-8 encoding set" -ForegroundColor Green
Write-Host "[READY] Lakehouse environment ready!" -ForegroundColor Cyan
Write-Host "[TIP] Try: lakehouse --help" -ForegroundColor Yellow

