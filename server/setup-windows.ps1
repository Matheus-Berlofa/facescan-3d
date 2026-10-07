$ErrorActionPreference = "Stop"
$ServerRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $ServerRoot ".venv"
$Requirements = Join-Path $ServerRoot "requirements.txt"

$Python = Get-Command py -ErrorAction SilentlyContinue
if (-not $Python) {
    throw "Python Launcher não encontrado. Instale Python 3.10 x64 e marque a opção 'py launcher'."
}

if (-not (Test-Path -LiteralPath $VenvRoot)) {
    & py -3.10 -m venv $VenvRoot
}

$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip wheel setuptools
& $VenvPython -m pip install -r $Requirements

Write-Host ""
Write-Host "Servidor base instalado." -ForegroundColor Green
Write-Host "Próximos passos:" -ForegroundColor Cyan
Write-Host "1. Instale o PyTorch CUDA compatível com a GTX 1060."
Write-Host "2. Coloque o repositório oficial DECA em server\engines\DECA."
Write-Host "3. Baixe os pesos licenciados seguindo as instruções oficiais do DECA/FLAME."
Write-Host "4. Copie .env.example para .env e configure a chave privada."
Write-Host "5. Execute run-server.ps1."

