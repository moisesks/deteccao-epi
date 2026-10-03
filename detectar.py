#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de EPI — capacete, colete de seguranca e cabeca descoberta.

Modelo treinado por Moises Kleinschmitt sobre o conjunto SH17, no ambito do
artigo "Deteccao automatica de equipamentos de protecao individual por visao
computacional" (Universidade Feevale, 2026).

Uso rapido:
    python detectar.py                          # webcam
    python detectar.py --fonte foto.jpg         # uma imagem
    python detectar.py --fonte pasta/           # todas as imagens da pasta
    python detectar.py --fonte video.mp4        # um video
    python detectar.py --fonte tela             # captura a tela do PC

Teclas durante a execucao: Q ou ESC encerra, ESPACO pausa, S salva o quadro.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EXT_IMG = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
EXT_VID = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.mpg', '.mpeg'}

# BGR. Verde = protegido, laranja = colete, vermelho = cabeca descoberta.
CORES = {
    'helmet':      (110, 190, 60),
    'safety-vest': (40, 130, 235),
    'head':        (60, 60, 220),
}
ROTULOS = {
    'helmet':      'capacete',
    'safety-vest': 'colete',
    'head':        'cabeca',
}


# --------------------------------------------------------------- argumentos
def ler_argumentos():
    p = argparse.ArgumentParser(
        description='Detector de EPI (capacete, colete, cabeca descoberta).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument('--fonte', default='webcam',
                   help='webcam | tela | caminho de imagem, video ou pasta | '
                        'URL RTSP. Padrao: webcam')
    p.add_argument('--pesos', default='pesos/yolo26n_best.pt',
                   help='arquivo .pt do modelo. Padrao: pesos/yolo26n_best.pt')
    p.add_argument('--conf', type=float, default=0.35,
                   help='confianca minima, de 0 a 1. Padrao: 0.35')
    p.add_argument('--iou', type=float, default=0.5,
                   help='limiar de IoU da supressao de sobrepostos. Padrao: 0.5')
    p.add_argument('--camera', type=int, default=0,
                   help='indice da webcam quando --fonte webcam. Padrao: 0')
    p.add_argument('--monitor', type=int, default=1,
                   help='numero do monitor quando --fonte tela. Padrao: 1')
    p.add_argument('--saida', default='saidas',
                   help='pasta onde gravar os resultados. Padrao: saidas')
    p.add_argument('--salvar', action='store_true',
                   help='grava o resultado em disco (automatico para imagem e pasta)')
    p.add_argument('--sem-janela', action='store_true',
                   help='nao abre janela; util para processar em lote')
    p.add_argument('--largura', type=int, default=1280,
                   help='largura da janela de exibicao. Padrao: 1280')
    return p.parse_args()


# --------------------------------------------------------------- desenho
def desenhar(quadro, resultado, nomes):
    """Desenha as caixas e devolve a contagem por classe."""
    contagem = {c: 0 for c in nomes.values()}
    caixas = resultado.boxes
    if caixas is None or len(caixas) == 0:
        return quadro, contagem

    for caixa in caixas:
        x1, y1, x2, y2 = (int(v) for v in caixa.xyxy[0])
        classe = nomes[int(caixa.cls[0])]
        conf = float(caixa.conf[0])
        contagem[classe] = contagem.get(classe, 0) + 1

        cor = CORES.get(classe, (200, 200, 200))
        espessura = 2 if max(quadro.shape[:2]) < 1000 else 3
        cv2.rectangle(quadro, (x1, y1), (x2, y2), cor, espessura)

        texto = f'{ROTULOS.get(classe, classe)} {conf:.2f}'
        escala = 0.5 if max(quadro.shape[:2]) < 1000 else 0.6
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
        topo = max(y1 - th - 8, 0)
        cv2.rectangle(quadro, (x1, topo), (x1 + tw + 8, topo + th + 8), cor, -1)
        cv2.putText(quadro, texto, (x1 + 4, topo + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return quadro, contagem


def painel(quadro, contagem, fps=None):
    """Faixa superior com a contagem por classe e um indicador de atencao."""
    h, w = quadro.shape[:2]
    altura = 64 if w >= 900 else 52
    faixa = quadro[0:altura, 0:w].copy()
    escuro = np.zeros_like(faixa)
    quadro[0:altura, 0:w] = cv2.addWeighted(faixa, 0.35, escuro, 0.65, 0)

    esc = 0.62 if w >= 900 else 0.5
    y = int(altura * 0.62)
    x = 14
    for classe in ('helmet', 'safety-vest', 'head'):
        n = contagem.get(classe, 0)
        cv2.circle(quadro, (x, y - 5), 6, CORES[classe], -1)
        txt = f'{ROTULOS[classe]}: {n}'
        cv2.putText(quadro, txt, (x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, esc,
                    (245, 245, 245), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, esc, 1)
        x += tw + 46

    # Indicador de atencao: ha cabeca descoberta no quadro.
    # ATENCAO: isto e uma leitura por QUADRO, nao por pessoa. O detector nao
    # associa o equipamento a um individuo — ver secao 4.8 do artigo.
    if contagem.get('head', 0) > 0:
        aviso = 'ATENCAO: cabeca encontrada no quadro'
        (tw, th), _ = cv2.getTextSize(aviso, cv2.FONT_HERSHEY_SIMPLEX, esc, 2)
        cv2.rectangle(quadro, (w - tw - 26, 8), (w - 8, 8 + th + 14),
                      (60, 60, 220), -1)
        cv2.putText(quadro, aviso, (w - tw - 17, 8 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, esc, (255, 255, 255), 2,
                    cv2.LINE_AA)
    elif fps is not None:
        txt = f'{fps:.1f} fps'
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, esc, 1)
        cv2.putText(quadro, txt, (w - tw - 16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    esc, (245, 245, 245), 1, cv2.LINE_AA)
    return quadro


def fechar_janelas(args=None):
    """Fecha as janelas sem quebrar em ambiente sem interface grafica."""
    if args is not None and getattr(args, 'sem_janela', False):
        return
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def redimensionar(quadro, largura):
    h, w = quadro.shape[:2]
    if w <= largura:
        return quadro
    return cv2.resize(quadro, (largura, int(h * largura / w)),
                      interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------- modos
def processar_imagens(modelo, caminhos, args, nomes):
    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    for caminho in caminhos:
        quadro = cv2.imread(str(caminho))
        if quadro is None:
            print(f'  ! nao consegui abrir {caminho.name}')
            continue
        r = modelo.predict(quadro, conf=args.conf, iou=args.iou, verbose=False)[0]
        quadro, contagem = desenhar(quadro, r, nomes)
        quadro = painel(quadro, contagem)

        destino = pasta / f'{caminho.stem}_detectado.jpg'
        cv2.imwrite(str(destino), quadro)
        resumo = ', '.join(f'{ROTULOS[c]}={contagem.get(c,0)}'
                           for c in ('helmet', 'safety-vest', 'head'))
        print(f'  {caminho.name}: {resumo}  ->  {destino}')

        if not args.sem_janela:
            cv2.imshow('EPI', redimensionar(quadro, args.largura))
            print('    (qualquer tecla avanca, Q encerra)')
            if cv2.waitKey(0) in (ord('q'), ord('Q'), 27):
                break
    fechar_janelas(args)


def processar_fluxo(modelo, captura, args, nomes, gravar_em=None, total=None):
    escritor = None
    if gravar_em is not None:
        Path(args.saida).mkdir(parents=True, exist_ok=True)
        fps_src = captura.get(cv2.CAP_PROP_FPS) or 25
        w = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        escritor = cv2.VideoWriter(str(gravar_em),
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps_src, (w, h))

    n, t0, fps, pausado = 0, time.time(), None, False
    try:
        while True:
            if not pausado:
                ok, quadro = captura.read()
                if not ok:
                    break
                r = modelo.predict(quadro, conf=args.conf, iou=args.iou,
                                   verbose=False)[0]
                quadro, contagem = desenhar(quadro, r, nomes)
                n += 1
                if n % 10 == 0:
                    fps = 10 / max(time.time() - t0, 1e-6)
                    t0 = time.time()
                quadro = painel(quadro, contagem, fps)
                ultimo = quadro
                if escritor is not None:
                    escritor.write(quadro)
                if total and n % 50 == 0:
                    print(f'  {n}/{total} quadros')

            if not args.sem_janela:
                cv2.imshow('EPI', redimensionar(ultimo, args.largura))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), ord('Q'), 27):
                    break
                if k == ord(' '):
                    pausado = not pausado
                if k in (ord('s'), ord('S')):
                    Path(args.saida).mkdir(parents=True, exist_ok=True)
                    destino = Path(args.saida) / f'quadro_{int(time.time())}.jpg'
                    cv2.imwrite(str(destino), ultimo)
                    print(f'  quadro salvo em {destino}')
    finally:
        captura.release()
        if escritor is not None:
            escritor.release()
            print(f'\nvideo gravado em {gravar_em}')
        fechar_janelas(args)
    print(f'{n} quadros processados')


def processar_tela(modelo, args, nomes):
    try:
        import mss
    except ImportError:
        print('Para capturar a tela instale a biblioteca mss:\n'
              '    pip install mss')
        sys.exit(1)

    print('Capturando a tela. Coloque o video ou a imagem em primeiro plano.')
    print('Q ou ESC encerra, S salva o quadro.\n')
    n, t0, fps = 0, time.time(), None
    with mss.mss() as sct:
        if args.monitor >= len(sct.monitors):
            print(f'Monitor {args.monitor} nao existe. '
                  f'Disponiveis: 1 a {len(sct.monitors) - 1}')
            sys.exit(1)
        area = sct.monitors[args.monitor]
        try:
            while True:
                bruto = np.array(sct.grab(area))
                quadro = cv2.cvtColor(bruto, cv2.COLOR_BGRA2BGR)
                r = modelo.predict(quadro, conf=args.conf, iou=args.iou,
                                   verbose=False)[0]
                quadro, contagem = desenhar(quadro, r, nomes)
                n += 1
                if n % 10 == 0:
                    fps = 10 / max(time.time() - t0, 1e-6)
                    t0 = time.time()
                quadro = painel(quadro, contagem, fps)
                cv2.imshow('EPI - captura de tela',
                           redimensionar(quadro, args.largura))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), ord('Q'), 27):
                    break
                if k in (ord('s'), ord('S')):
                    Path(args.saida).mkdir(parents=True, exist_ok=True)
                    destino = Path(args.saida) / f'tela_{int(time.time())}.jpg'
                    cv2.imwrite(str(destino), quadro)
                    print(f'  quadro salvo em {destino}')
        finally:
            fechar_janelas(args)
    print(f'{n} quadros processados')


# --------------------------------------------------------------- principal
def main():
    args = ler_argumentos()

    pesos = Path(args.pesos)
    if not pesos.exists():
        print(f'Nao encontrei o arquivo de pesos: {pesos}')
        print('Verifique se a pasta "pesos" esta ao lado deste script, ou use '
              '--pesos caminho/para/modelo.pt')
        sys.exit(1)

    from ultralytics import YOLO          # importado aqui: demora alguns segundos
    print(f'Carregando {pesos.name} ...')
    modelo = YOLO(str(pesos))
    nomes = modelo.names
    print(f'Classes do modelo: {", ".join(nomes.values())}')
    print(f'Confianca minima: {args.conf}\n')

    fonte = args.fonte

    if fonte == 'tela':
        processar_tela(modelo, args, nomes)
        return

    if fonte == 'webcam':
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW
                               if sys.platform == 'win32' else 0)
        if not cap.isOpened():
            print(f'Nao consegui abrir a webcam {args.camera}. '
                  f'Tente --camera 1.')
            sys.exit(1)
        print('Webcam aberta. Q ou ESC encerra, ESPACO pausa, S salva.\n')
        gravar = Path(args.saida) / 'webcam.mp4' if args.salvar else None
        processar_fluxo(modelo, cap, args, nomes, gravar)
        return

    caminho = Path(fonte)

    if caminho.is_dir():
        imagens = sorted(p for p in caminho.iterdir()
                         if p.suffix.lower() in EXT_IMG)
        if not imagens:
            print(f'Nenhuma imagem encontrada em {caminho}')
            sys.exit(1)
        print(f'{len(imagens)} imagens encontradas.\n')
        processar_imagens(modelo, imagens, args, nomes)
        return

    if caminho.is_file():
        if caminho.suffix.lower() in EXT_IMG:
            processar_imagens(modelo, [caminho], args, nomes)
            return
        if caminho.suffix.lower() in EXT_VID:
            cap = cv2.VideoCapture(str(caminho))
            if not cap.isOpened():
                print(f'Nao consegui abrir o video {caminho}')
                sys.exit(1)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            gravar = Path(args.saida) / f'{caminho.stem}_detectado.mp4'
            print(f'Processando {caminho.name}'
                  + (f' ({total} quadros)' if total else '') + '\n')
            processar_fluxo(modelo, cap, args, nomes, gravar, total)
            return
        print(f'Extensao nao reconhecida: {caminho.suffix}')
        sys.exit(1)

    # RTSP, HTTP ou indice numerico de camera
    cap = cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)
    if not cap.isOpened():
        print(f'Nao consegui abrir a fonte: {fonte}')
        sys.exit(1)
    gravar = Path(args.saida) / 'fluxo.mp4' if args.salvar else None
    processar_fluxo(modelo, cap, args, nomes, gravar)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nEncerrado pelo usuario.')
