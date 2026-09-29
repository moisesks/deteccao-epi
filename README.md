# Detector de EPI

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO26n%20%7C%20YOLO11n-111F68)
![Licença](https://img.shields.io/badge/licen%C3%A7a-AGPL--3.0-blue)
![Plataforma](https://img.shields.io/badge/plataforma-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

Detecta **capacete**, **colete de segurança** e **cabeça descoberta** em imagens,
vídeos, webcam, câmeras IP ou na tela do computador.

Os modelos incluídos foram treinados por Moisés Kleinschmitt Da Silva sobre o
conjunto [SH17](https://arxiv.org/abs/2407.04590), no âmbito do artigo
*Detecção automática de equipamentos de proteção individual por visão
computacional em canteiros de obra e ambientes industriais* (Universidade
Feevale, 2026).

![Exemplo de detecção](exemplos/amostras_sh17.png)

## Desempenho

Medido na partição de teste do SH17:

| modelo | mAP50 | capacete | colete | cabeça | fps (T4) |
|---|---|---|---|---|---|
| `yolo26n_best.pt` (padrão) | 0,6154 | 0,6237 | 0,3466 | 0,8760 | 78,7 |
| `yolo11n_best.pt` | 0,5956 | 0,5814 | 0,3429 | 0,8627 | 79,6 |

---

## Instalação

### Windows

1. Instale o **Python 3.10 ou superior** de https://www.python.org/downloads/,
   marcando **"Add Python to PATH"** na primeira tela.
2. Baixe este repositório (**Code → Download ZIP**, ou `git clone`).
3. Dê dois cliques em **`instalar.bat`**.

A instalação baixa cerca de 2 GB na primeira vez, porque o PyTorch vem junto.
Depois disso funciona offline.

### Linux ou macOS

```bash
git clone https://github.com/moisesks/deteccao-epi.git
cd deteccao-epi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Uso

Dois cliques nos atalhos, ou pela linha de comando:

| O que você quer | Atalho | Comando |
|---|---|---|
| Webcam ao vivo | `webcam.bat` | `python detectar.py` |
| Capturar a tela | `tela.bat` | `python detectar.py --fonte tela` |
| Imagens de exemplo | `exemplos.bat` | `python detectar.py --fonte exemplos` |
| Uma foto | — | `python detectar.py --fonte foto.jpg` |
| Uma pasta de fotos | — | `python detectar.py --fonte C:\fotos` |
| Um vídeo | — | `python detectar.py --fonte video.mp4` |
| Câmera IP | — | `python detectar.py --fonte rtsp://usuario:senha@192.168.0.50/stream` |

No **PowerShell**, chame o Python do ambiente diretamente — não precisa ativar:

```powershell
.venv\Scripts\python.exe detectar.py --fonte tela
```

**Modo tela** é o mais prático para testar: abra um vídeo do YouTube de obra ou
fábrica, rode `tela.bat`, e a detecção acontece sobre o que estiver na tela.
Deixe a janela de resultado em outro monitor (ou pequena num canto) para não
criar um efeito de espelho.

### Teclas durante a execução

| Tecla | Ação |
|---|---|
| `Q` ou `ESC` | encerra |
| `ESPAÇO` | pausa e retoma (webcam e vídeo) |
| `S` | salva o quadro atual em `saidas/` |

### Onde ficam os resultados

Tudo vai para a pasta **`saidas/`**: imagens anotadas, vídeos processados
(`<nome>_detectado.mp4`) e os quadros salvos com `S`.

---

## Opções

| Opção | O que faz | Padrão |
|---|---|---|
| `--conf 0.35` | confiança mínima, de 0 a 1 | 0.35 |
| `--iou 0.5` | limiar de sobreposição entre caixas | 0.5 |
| `--pesos pesos/yolo11n_best.pt` | troca de modelo | yolo26n |
| `--camera 1` | escolhe outra webcam | 0 |
| `--monitor 2` | escolhe outro monitor no modo tela | 1 |
| `--largura 1600` | largura da janela | 1280 |
| `--salvar` | grava o vídeo da webcam ou câmera IP | desligado |
| `--sem-janela` | processa sem abrir janela | desligado |
| `--saida C:\resultados` | outra pasta de saída | saidas |

**Ajuste que mais muda o resultado:** o `--conf`. Com 0.35 o detector é
equilibrado. Baixe para **0.20** se ele estiver deixando passar coisas; suba para
**0.50** se estiver marcando demais.

---

## Como interpretar

As caixas são coloridas por classe: **verde** capacete, **laranja** colete,
**vermelho** cabeça descoberta. A faixa no topo conta quantos de cada tipo há no
quadro.

O aviso **"ATENÇÃO: cabeça descoberta no quadro"** aparece quando alguma cabeça
sem capacete é detectada. É uma leitura **por quadro, não por pessoa**: o
detector localiza os objetos, mas não associa cada capacete ao seu respectivo
trabalhador. Uma cena com duas pessoas, uma de capacete e outra sem, acende o
aviso — o que está correto — mas uma cena com um capacete apoiado sobre uma
bancada e ninguém usando também não acusaria irregularidade. Essa limitação está
declarada na seção 4.8 do artigo; o caminho para resolvê-la é acrescentar
estimativa de pose, associando cada equipamento ao esqueleto da pessoa mais
próxima.

### O que esperar de cada classe

- **Cabeça descoberta** é a classe mais confiável, mAP50 de 0,876.
- **Capacete** é intermediário, 0,624.
- **Colete** é o ponto fraco, 0,347. O modelo deixa de sinalizar cerca de dois
  terços dos coletes presentes. A causa é o desequilíbrio do conjunto de
  treinamento, com apenas 311 instâncias de colete contra 7.135 de cabeça.

As imagens do SH17 vêm de banco de fotografias profissionais. Em imagens de
câmera de segurança real, com iluminação ruim e pessoas pequenas no quadro, o
desempenho cai. Bonés comuns podem ser confundidos com capacete.

---

## Problemas comuns

**"Não consegui abrir a webcam 0"** — tente `--camera 1`. Feche o Teams, o Zoom
ou qualquer programa que esteja usando a câmera.

**"a execução de scripts foi desabilitada neste sistema"** ao rodar
`.venv\Scripts\activate` no PowerShell — use `.venv\Scripts\python.exe` direto,
ou libere scripts locais para o seu usuário:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Os `.bat` não são afetados.

**"Could not find a suitable TLS CA certificate bundle"** no `pip install` —
alguma variável de ambiente (`CURL_CA_BUNDLE`, `REQUESTS_CA_BUNDLE` ou
`SSL_CERT_FILE`) aponta para um arquivo que não existe, geralmente sobra de outro
programa instalado. Verifique com
`Get-ChildItem Env: | Where-Object { $_.Name -match 'CA_BUNDLE|SSL_CERT' }` e
remova a variável em *Propriedades do Sistema → Variáveis de Ambiente*.

**A janela abre preta ou trava no modo tela** — experimente `--monitor 2`.

**Muito lento** — sem placa de vídeo NVIDIA o modelo roda na CPU, entre 3 e 10
quadros por segundo. Para imagens e vídeos isso não importa; para webcam,
incomoda. Com uma GPU NVIDIA, instale a versão CUDA do PyTorch conforme
https://pytorch.org/get-started/locally/.

**"ModuleNotFoundError: mss"** — só afeta o modo tela. Rode `pip install mss`
com o ambiente ativo.

---

## Estrutura

```
detectar.py        script principal
pesos/             modelos treinados (yolo26n_best.pt, yolo11n_best.pt)
exemplos/          imagem de amostra do SH17
instalar.bat       cria o ambiente e instala as dependências (Windows)
webcam.bat         atalhos de execução
tela.bat
exemplos.bat
requirements.txt
```

---

## Como citar

Se este detector ou os modelos forem úteis no seu trabalho, cite o artigo:

```bibtex
@misc{kleinschmitt2026epi,
  author      = {Kleinschmitt Da Silva, Moisés},
  title       = {Detecção automática de equipamentos de proteção individual por
                 visão computacional em canteiros de obra e ambientes industriais},
  institution = {Universidade Feevale},
  year        = {2026},
  howpublished = {\url{https://github.com/moisesks/deteccao-epi}}
}
```

E o conjunto de dados utilizado no treinamento:

> Ahmad, H. M.; Rahimi, A. *SH17: A Dataset for Human Safety and Personal
> Protective Equipment Detection in Manufacturing Industry*. arXiv:2407.04590, 2024.

---

## Licença

Distribuído sob a [GNU AGPL-3.0](LICENSE), herdada do
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics). Uso acadêmico e
pessoal sem restrição. Para uso comercial ou como serviço acessado pela rede, a
AGPL exige a disponibilização do código-fonte das modificações.
