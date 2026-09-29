# Detector de EPI

Detecta **capacete**, **colete de segurança** e **cabeça descoberta** em imagens,
vídeos, webcam ou na tela do computador.

Os modelos aqui incluídos foram treinados por Moisés Kleinschmitt sobre o conjunto
SH17, no âmbito do artigo *Detecção automática de equipamentos de proteção
individual por visão computacional em canteiros de obra e ambientes industriais*
(Universidade Feevale, 2026). Desempenho medido na partição de teste:

| modelo | mAP50 | capacete | colete | cabeça | fps (T4) |
|---|---|---|---|---|---|
| `yolo26n_best.pt` (padrão) | 0,6154 | 0,6237 | 0,3466 | 0,8760 | 78,7 |
| `yolo11n_best.pt` | 0,5956 | 0,5814 | 0,3429 | 0,8627 | 79,6 |

---

## Instalação (Windows)

1. Instale o **Python 3.10 ou superior** de https://www.python.org/downloads/,
   marcando a opção **"Add Python to PATH"** na primeira tela.
2. Dê dois cliques em **`instalar.bat`**.

A instalação baixa cerca de 2 GB na primeira vez, porque o PyTorch vem junto.
Depois disso não baixa mais nada e funciona offline.

No Linux ou macOS, no terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Uso

Dois cliques nos atalhos, ou pela linha de comando com o ambiente ativo
(`.venv\Scripts\activate` no Windows):

| O que você quer | Atalho | Comando |
|---|---|---|
| Webcam ao vivo | `webcam.bat` | `python detectar.py` |
| Capturar a tela | `tela.bat` | `python detectar.py --fonte tela` |
| Imagens de exemplo | `exemplos.bat` | `python detectar.py --fonte exemplos` |
| Uma foto | — | `python detectar.py --fonte foto.jpg` |
| Uma pasta de fotos | — | `python detectar.py --fonte C:\fotos` |
| Um vídeo | — | `python detectar.py --fonte video.mp4` |
| Câmera IP | — | `python detectar.py --fonte rtsp://usuario:senha@192.168.0.50/stream` |

**Modo tela** é o mais prático para testar: abra um vídeo do YouTube ou uma foto
em tela cheia, rode `tela.bat`, e a detecção acontece por cima do que estiver
aparecendo. Não precisa baixar nada.

### Teclas durante a execução

| Tecla | Ação |
|---|---|
| `Q` ou `ESC` | encerra |
| `ESPAÇO` | pausa e retoma |
| `S` | salva o quadro atual em `saidas/` |

### Onde ficam os resultados

Tudo vai para a pasta **`saidas/`**: imagens anotadas, vídeos processados e os
quadros que você salvar com `S`.

---

## Opções

| Opção | O que faz | Padrão |
|---|---|---|
| `--conf 0.35` | confiança mínima, de 0 a 1 | 0.35 |
| `--pesos pesos/yolo11n_best.pt` | troca de modelo | yolo26n |
| `--camera 1` | escolhe outra webcam | 0 |
| `--monitor 2` | escolhe outro monitor no modo tela | 1 |
| `--largura 1600` | tamanho da janela | 1280 |
| `--salvar` | grava o vídeo da webcam | desligado |
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
declarada na seção 4.8 do artigo, e o caminho para resolvê-la é acrescentar
estimativa de pose, associando cada equipamento ao esqueleto da pessoa mais
próxima.

## O que esperar de cada classe

O desempenho não é uniforme, e conhecer isso evita interpretar mal o que você vê:

- **Cabeça descoberta** é a classe mais confiável, mAP50 de 0,876.
- **Capacete** é intermediário, 0,624.
- **Colete** é o ponto fraco, 0,347. O modelo deixa de sinalizar cerca de dois
  terços dos coletes presentes. A causa é o desequilíbrio do conjunto de
  treinamento, que tinha apenas 311 instâncias de colete contra 7.135 de cabeça.

As imagens do SH17 vêm de banco de fotografias profissionais. Em imagens de
câmera de segurança real, com iluminação ruim e pessoas pequenas no quadro, o
desempenho cai.

---

## Problemas comuns

**"Não consegui abrir a webcam 0"** — tente `python detectar.py --camera 1`.
Feche o Teams, o Zoom ou qualquer programa que esteja usando a câmera.

**A janela abre preta ou trava** — no modo tela, verifique o monitor com
`--monitor 2`.

**Muito lento** — sem placa de vídeo NVIDIA o modelo roda na CPU, entre 3 e 10
quadros por segundo. Funciona, mas fica travado. Para imagens e vídeos isso não
importa; para webcam, incomoda. Com uma GPU NVIDIA, instale a versão CUDA do
PyTorch conforme https://pytorch.org/get-started/locally/.

**"ModuleNotFoundError: mss"** — só afeta o modo tela. Rode
`pip install mss` com o ambiente ativo.

---

## Licença

Os modelos derivam do Ultralytics YOLO, distribuído sob **AGPL-3.0**. Uso
acadêmico e pessoal sem restrição. Para uso comercial ou como serviço em rede, a
AGPL exige a abertura do código-fonte — consulte o jurídico antes.
