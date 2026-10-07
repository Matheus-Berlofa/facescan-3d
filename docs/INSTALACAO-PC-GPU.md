# Instalação no PC da GPU

Configuração-alvo: Windows 10/11, NVIDIA GTX 1060 6 GB, driver atualizado e conexão pelo Parsec.

## 1. Instalar a base do servidor

Instale Python 3.10 x64 e Git. No PowerShell, dentro da pasta `server`, execute:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-windows.ps1
```

Esse comando cria `server\.venv` e instala somente a API e as bibliotecas leves.

## 2. Instalar CUDA para Python

A versão do CUDA mostrada pelo `nvidia-smi` é a capacidade máxima do driver, não a versão que precisa ser instalada manualmente. Para a GTX 1060, use uma versão do PyTorch que ainda ofereça suporte à arquitetura Pascal. A instalação deve ser validada com:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

O resultado precisa mostrar `True` e `NVIDIA GeForce GTX 1060 6GB`.

## 3. Preparar DECA e FLAME

Clone o repositório oficial DECA em `server\engines\DECA` e siga o procedimento oficial de obtenção dos dados. Ao final, precisam existir:

```text
server/engines/DECA/decalib/deca.py
server/engines/DECA/data/deca_model.tar
server/engines/DECA/data/generic_model.pkl
```

Para textura estimada pelo modelo, adicione também:

```text
server/engines/DECA/data/FLAME_albedo_from_BFM.npz
```

Os arquivos não são redistribuídos neste projeto porque DECA e FLAME possuem termos próprios. O DECA oficial está em <https://github.com/yfeng95/DECA> e o FLAME em <https://flame.is.tue.mpg.de/>.

## 4. Configurar a chave privada

Copie `.env.example` para `.env`. Gere uma chave longa:

```powershell
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
[Convert]::ToBase64String($bytes)
```

Cole o resultado em `FACESCAN_TOKEN`. Ajuste `DECA_ROOT` para a pasta real do DECA.

## 5. Iniciar servidor e túnel HTTPS

Inicie a API:

```powershell
.\run-server.ps1
```

Em outro PowerShell, depois de instalar `cloudflared`, abra um túnel temporário:

```powershell
cloudflared tunnel --url http://127.0.0.1:8765
```

O Cloudflare exibirá um endereço parecido com:

```text
https://exemplo-aleatorio.trycloudflare.com
```

No site, use:

```text
wss://exemplo-aleatorio.trycloudflare.com/ws/scan
```

e a mesma chave configurada em `FACESCAN_TOKEN`.

## 6. Verificação

Antes de capturar o rosto, abra no PC da GPU:

```text
http://127.0.0.1:8765/health
```

O JSON deve informar `"engineReady": true`. Caso contrário, a mensagem indicará o arquivo ou componente ausente.

## Privacidade

- Os JPEGs recebidos nunca são escritos em arquivo pelo servidor.
- As referências aos buffers são descartadas ao concluir, cancelar ou desconectar, permitindo que a memória seja recuperada pelo processo.
- O ZIP derivado expira após 30 minutos.
- O site não inclui a chave no repositório e a mantém apenas durante a sessão do navegador.
- Encerre `cloudflared` e o servidor quando terminar.

