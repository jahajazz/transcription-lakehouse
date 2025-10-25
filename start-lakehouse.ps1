# Start the transcription lakehouse environment with UTF-8 encoding
$env:PYTHONIOENCODING="utf-8"

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "❌ Virtual environment not found. Run 'python -m venv venv' first." -ForegroundColor Red
    exit 1
}

Write-Host "✅ UTF-8 encoding set" -ForegroundColor Green
Write-Host "🚀 Lakehouse environment ready!" -ForegroundColor Cyan
Write-Host "💡 Try: lakehouse --help" -ForegroundColor Yellow

