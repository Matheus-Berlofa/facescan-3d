$ErrorActionPreference = "Stop"
$ServerRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ServerRoot ".venv\Scripts\python.exe"
$EnvFile = Join-Path $ServerRoot ".env"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Ambiente virtual ausente. Execute setup-windows.ps1 primeiro."
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Arquivo .env ausente. Copie .env.example para .env e configure a chave e o DECA_ROOT."
}

Set-Location -LiteralPath $ServerRoot
& $VenvPython -m uvicorn app:app --host 127.0.0.1 --port 8765

