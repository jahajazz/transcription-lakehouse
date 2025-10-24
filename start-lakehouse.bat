@echo off
REM Start PowerShell with UTF-8 encoding for the transcription lakehouse project
cd /d "%~dp0"
powershell.exe -Command "& {$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\Activate.ps1; Write-Host 'Lakehouse environment ready! UTF-8 encoding set.' -ForegroundColor Green; Write-Host 'Virtual environment activated.' -ForegroundColor Green; Write-Host 'You can now use: lakehouse --help' -ForegroundColor Cyan}"
