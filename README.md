# FaceScan 3D Live

Captura facial guiada no navegador e reconstrução neural em uma GPU privada. O projeto foi desenhado para gerar uma malha base em OBJ/GLB que possa ser refinada e adaptada ao PES 2021.

## Arquitetura

- `dist/`: site estático publicado pelo GitHub Pages.
- `server/`: processador privado executado no PC com NVIDIA.
- O navegador usa MediaPipe apenas para orientação, cobertura e qualidade.
- Quadros JPEG de 512 × 512 são transmitidos por WebSocket seguro (`wss://`).
- O servidor mantém até 18 quadros somente em memória e os apaga após o processamento.
- DECA estima detalhe e aparência; FLAME fornece uma topologia facial consistente.
- Apenas os arquivos derivados ficam disponíveis para download, por 30 minutos por padrão.

> “Sem salvar fotos” não significa “sem transmitir pixels”. A GPU precisa receber os quadros para reconstruir o rosto. O projeto evita persistência das imagens originais e exige WSS e chave de acesso.

## Instalação

Consulte [docs/INSTALACAO-PC-GPU.md](docs/INSTALACAO-PC-GPU.md). Os pesos DECA/FLAME têm licenças próprias e não são incluídos neste repositório.

## Desenvolvimento do site

Sirva a pasta `dist` por HTTP local. A câmera não funciona quando o arquivo HTML é aberto diretamente pelo Explorer.

```powershell
node ..\static-preview-server.mjs
```

O GitHub Actions publica `dist/` automaticamente quando a branch `main` recebe alterações.

## Uso responsável

O rosto é dado biométrico sensível. Use uma chave longa, não compartilhe o endereço do túnel e encerre o servidor quando terminar. Este projeto é para uso pessoal e experimental. Confirme as licenças dos modelos antes de qualquer uso comercial.

