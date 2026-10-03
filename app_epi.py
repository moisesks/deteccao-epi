#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de EPI — interface gráfica (PySide6 / Qt).

Mesmos modelos e mesmo desenho de caixas do detectar.py, com uma janela de
controle: escolha da fonte, do modelo e da confiança, contadores por classe,
alerta de cabeça descoberta, pausa, captura de quadro e gravação.

Uso:
    python app_epi.py            (ou dois cliques em app.bat)
    python app_epi.py --print    abre com as imagens de exemplo, salva a janela
                                 em exemplos/interface.png e fecha sozinho

Atalhos: ESPAÇO pausa, S salva o quadro, ESC para, F11 tela cheia,
F12 salva uma imagem da janela inteira (em saidas/).
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QRectF
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QImage, QKeySequence,
                           QPainter, QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget, QVBoxLayout,
    QWidget,
)

from detectar import EXT_IMG, EXT_VID

BASE = Path(__file__).resolve().parent
PASTA_PESOS = BASE / 'pesos'
PASTA_SAIDA = BASE / 'saidas'
PASTA_EXEMPLOS = BASE / 'exemplos'

# Cores em RGB (as mesmas do detectar.py, que usa BGR).
CLASSES = [
    ('helmet',      'Capacete',          '#3CBE6E'),
    ('safety-vest', 'Colete',            '#EB8228'),
    ('head',        'Cabeça',            '#DC3C3C'),
]

FONTES = [
    ('webcam',   'Webcam'),
    ('tela',     'Captura de tela'),
    ('video',    'Arquivo de vídeo'),
    ('imagens',  'Imagem ou pasta de imagens'),
    ('rtsp',     'Câmera IP (RTSP / HTTP)'),
    ('exemplos', 'Imagens de exemplo'),
]

_MODELOS = {}


def carregar_modelo(caminho):
    """Carrega o modelo uma vez e reaproveita nas execuções seguintes."""
    chave = str(caminho)
    if chave not in _MODELOS:
        from ultralytics import YOLO  # importado aqui: demora alguns segundos
        _MODELOS[chave] = YOLO(chave)
    return _MODELOS[chave]


COR = {chave: QColor(cor) for chave, _, cor in CLASSES}
ROTULO = {chave: rotulo for chave, rotulo, _ in CLASSES}


def para_qimage(quadro_bgr):
    rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def para_bgr(img):
    """QImage -> array BGR (para gravar vídeo e salvar com OpenCV)."""
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h, linha = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8, count=h * linha).reshape(h, linha)
    rgb = arr[:, :w * 3].reshape(h, w, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def anotar(quadro_bgr, resultado, nomes):
    """Desenha caixas e etiquetas (com acentos) e devolve (QImage, contagem)."""
    img = para_qimage(quadro_bgr)
    contagem = {c: 0 for c in nomes.values()}
    caixas = resultado.boxes
    if caixas is None or len(caixas) == 0:
        return img, contagem

    w = img.width()
    traco = max(2.0, w / 480)
    fonte = QFont('Segoe UI')
    fonte.setPixelSize(int(max(12, min(28, w / 75))))
    fonte.setBold(True)

    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setFont(fonte)
    fm = p.fontMetrics()
    for caixa in caixas:
        x1, y1, x2, y2 = (float(v) for v in caixa.xyxy[0])
        classe = nomes[int(caixa.cls[0])]
        conf = float(caixa.conf[0])
        contagem[classe] = contagem.get(classe, 0) + 1
        cor = COR.get(classe, QColor('#C8C8C8'))

        p.setPen(QPen(cor, traco))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(x1, y1, x2 - x1, y2 - y1), traco * 1.5, traco * 1.5)

        texto = f'{ROTULO.get(classe, classe)}  {conf:.0%}'
        tw, th = fm.horizontalAdvance(texto) + 12, fm.height() + 6
        topo = y1 - th - traco if y1 - th - traco > 0 else y1 + traco
        esquerda = max(0.0, min(x1 - traco / 2, w - tw))  # não sair pela borda
        etiqueta = QRectF(esquerda, topo, tw, th)
        p.setPen(Qt.NoPen)
        p.setBrush(cor)
        p.drawRoundedRect(etiqueta, 4, 4)
        p.setPen(QColor('#FFFFFF'))
        p.drawText(etiqueta, Qt.AlignCenter, texto)
    p.end()
    return img, contagem


# =================================================================== worker
class Detector(QThread):
    """Lê a fonte, roda o modelo e devolve quadros anotados para a janela."""

    quadro = Signal(QImage, dict, float)        # imagem, contagem, fps
    imagem = Signal(QImage, dict, str)          # modo imagens: um resultado por arquivo
    status = Signal(str)
    erro = Signal(str)

    def __init__(self, tipo, valor, pesos, conf, camera, monitor, gravar):
        super().__init__()
        self.tipo = tipo
        self.valor = valor
        self.pesos = pesos
        self.conf = conf
        self.iou = 0.5
        self.camera = camera
        self.monitor = monitor
        self.gravar = gravar
        self.pausado = False
        self._parar = False
        self._salvar = False
        self._escritor = None
        self._arquivo_gravacao = None
        self._ultimo = None

    # ------------------------------------------------ comandos da janela
    def parar(self):
        self._parar = True

    def alternar_pausa(self):
        self.pausado = not self.pausado
        return self.pausado

    def salvar_quadro(self):
        self._salvar = True

    # ------------------------------------------------ execução
    def run(self):
        try:
            self.status.emit(f'Carregando {Path(self.pesos).name} ...')
            modelo = carregar_modelo(self.pesos)
            nomes = modelo.names
            PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

            if self.tipo in ('imagens', 'exemplos'):
                self._rodar_imagens(modelo, nomes)
            elif self.tipo == 'tela':
                self._rodar_tela(modelo, nomes)
            else:
                self._rodar_captura(modelo, nomes)
        except Exception as e:  # mostra o erro na janela em vez de fechar
            self.erro.emit(f'{type(e).__name__}: {e}')
        finally:
            self._fechar_gravacao()

    def _detectar(self, modelo, nomes, quadro):
        r = modelo.predict(quadro, conf=self.conf, iou=self.iou, verbose=False)[0]
        return anotar(quadro, r, nomes)

    def _rodar_imagens(self, modelo, nomes):
        caminho = Path(self.valor)
        if caminho.is_dir():
            arquivos = sorted(p for p in caminho.iterdir() if p.suffix.lower() in EXT_IMG)
            if self.tipo == 'exemplos':  # o print da própria interface não é exemplo
                arquivos = [p for p in arquivos if p.name != 'interface.png']
        else:
            arquivos = [caminho]
        if not arquivos:
            self.erro.emit(f'Nenhuma imagem encontrada em {caminho}')
            return
        for i, arq in enumerate(arquivos, 1):
            if self._parar:
                break
            quadro = cv2.imread(str(arq))
            if quadro is None:
                self.status.emit(f'Não consegui abrir {arq.name}')
                continue
            img, contagem = self._detectar(modelo, nomes, quadro)
            destino = PASTA_SAIDA / f'{arq.stem}_detectado.jpg'
            img.save(str(destino), quality=92)
            self.imagem.emit(img, contagem, arq.name)
            self.status.emit(f'{i}/{len(arquivos)} imagens processadas — salvas em saidas/')

    def _rodar_captura(self, modelo, nomes):
        if self.tipo == 'webcam':
            api = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera, api)
            descricao = f'webcam {self.camera}'
        else:
            cap = cv2.VideoCapture(self.valor)
            descricao = Path(self.valor).name if self.tipo == 'video' else self.valor
        if not cap.isOpened():
            dica = ' Tente outro índice de câmera e feche Teams/Zoom.' if self.tipo == 'webcam' else ''
            self.erro.emit(f'Não consegui abrir {descricao}.{dica}')
            return

        fps_fonte = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps_fonte <= 1 or fps_fonte > 120:
            fps_fonte = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.tipo == 'video' else 0
        intervalo = 1.0 / fps_fonte if self.tipo == 'video' else 0.0

        self.status.emit(f'Rodando: {descricao}')
        n, t0, fps = 0, time.time(), 0.0
        try:
            while not self._parar:
                if self.pausado:
                    self._atender_salvar()
                    self.msleep(30)
                    continue
                inicio = time.time()
                ok, quadro = cap.read()
                if not ok:
                    if self.tipo == 'video':
                        self.status.emit(f'Vídeo concluído — {n} quadros processados')
                    else:
                        self.erro.emit('A fonte parou de enviar imagens.')
                    break
                img, contagem = self._detectar(modelo, nomes, quadro)
                n += 1
                if n % 10 == 0:
                    fps = 10 / max(time.time() - t0, 1e-6)
                    t0 = time.time()
                self._ultimo = img
                self._escrever(img, fps_fonte)
                self._atender_salvar()
                self.quadro.emit(img, contagem, fps)
                if total and n % 25 == 0:
                    self.status.emit(f'{descricao}: {n}/{total} quadros')
                # vídeo em arquivo: não passar mais rápido que o original
                if intervalo:
                    resto = intervalo - (time.time() - inicio)
                    if resto > 0:
                        time.sleep(resto)
        finally:
            cap.release()

    def _rodar_tela(self, modelo, nomes):
        try:
            import mss
        except ImportError:
            self.erro.emit('Para capturar a tela instale a biblioteca mss:  pip install mss')
            return
        n, t0, fps = 0, time.time(), 0.0
        with mss.mss() as sct:
            if self.monitor >= len(sct.monitors):
                self.erro.emit(f'Monitor {self.monitor} não existe. '
                               f'Disponíveis: 1 a {len(sct.monitors) - 1}')
                return
            area = sct.monitors[self.monitor]
            self.status.emit(f'Capturando o monitor {self.monitor} '
                             f'({area["width"]}x{area["height"]})')
            while not self._parar:
                if self.pausado:
                    self._atender_salvar()
                    self.msleep(30)
                    continue
                bruto = np.array(sct.grab(area))
                quadro = cv2.cvtColor(bruto, cv2.COLOR_BGRA2BGR)
                img, contagem = self._detectar(modelo, nomes, quadro)
                n += 1
                if n % 10 == 0:
                    fps = 10 / max(time.time() - t0, 1e-6)
                    t0 = time.time()
                self._ultimo = img
                self._escrever(img, 10.0)
                self._atender_salvar()
                self.quadro.emit(img, contagem, fps)

    # ------------------------------------------------ gravação e captura
    def _escrever(self, img, fps):
        if not self.gravar:
            self._fechar_gravacao()
            return
        quadro = para_bgr(img)
        if self._escritor is None:
            h, w = quadro.shape[:2]
            carimbo = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._arquivo_gravacao = PASTA_SAIDA / f'gravacao_{carimbo}.mp4'
            self._escritor = cv2.VideoWriter(str(self._arquivo_gravacao),
                                             cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (w, h))
            self.status.emit(f'Gravando em {self._arquivo_gravacao.name}')
        self._escritor.write(quadro)

    def _fechar_gravacao(self):
        if self._escritor is not None:
            self._escritor.release()
            self._escritor = None
            self.status.emit(f'Vídeo salvo: saidas/{self._arquivo_gravacao.name}')

    def _atender_salvar(self):
        if self._salvar and self._ultimo is not None:
            self._salvar = False
            carimbo = datetime.now().strftime('%Y%m%d_%H%M%S')
            destino = PASTA_SAIDA / f'quadro_{carimbo}.jpg'
            self._ultimo.save(str(destino), quality=92)
            self.status.emit(f'Quadro salvo: saidas/{destino.name}')


# =================================================================== widgets
class Video(QLabel):
    """Mostra o quadro mantendo a proporção ao redimensionar."""

    def __init__(self):
        super().__init__()
        self.setObjectName('video')
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img = None
        self.limpar()

    def limpar(self):
        self._img = None
        self.setPixmap(QPixmap())
        self.setText('Escolha a fonte ao lado e clique em Iniciar.')

    def mostrar(self, img):
        self._img = img
        self._ajustar()

    def _ajustar(self):
        if self._img is None:
            return
        pix = QPixmap.fromImage(self._img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._ajustar()


class Contador(QFrame):
    def __init__(self, rotulo, cor):
        super().__init__()
        self.setObjectName('cartao')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        faixa = QFrame()
        faixa.setFixedSize(12, 12)
        faixa.setStyleSheet(f'background:{cor};')
        nome = QLabel(rotulo)
        nome.setObjectName('cartaoNome')
        self.valor = QLabel('0')
        self.valor.setObjectName('cartaoValor')
        self.valor.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(faixa)
        lay.addSpacing(6)
        lay.addWidget(nome, 1)
        lay.addWidget(self.valor)

    def definir(self, n):
        self.valor.setText(str(n))


def titulo_secao(texto):
    lbl = QLabel(texto)
    lbl.setObjectName('secao')
    return lbl


# =================================================================== janela
class Janela(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Detector de EPI')
        self.resize(1360, 820)
        self.worker = None
        self.galeria = []
        self.indice = 0

        central = QWidget()
        self.setCentralWidget(central)
        raiz = QHBoxLayout(central)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(0)

        # ---------------- área principal
        principal = QWidget()
        principal.setObjectName('principal')
        lp = QVBoxLayout(principal)
        lp.setContentsMargins(12, 12, 12, 8)
        lp.setSpacing(8)

        self.alerta = QLabel()
        self.alerta.setObjectName('alerta')
        self.alerta.setAlignment(Qt.AlignCenter)
        self.alerta.setToolTip('Leitura por quadro, não por pessoa: o detector não associa\n'
                               'cada capacete a um trabalhador (seção 4.8 do artigo).')
        lp.addWidget(self.alerta)

        self.video = Video()
        lp.addWidget(self.video, 1)

        self.nav = QWidget()
        ln = QHBoxLayout(self.nav)
        ln.setContentsMargins(0, 0, 0, 0)
        self.btn_ant = QPushButton('Anterior')
        self.btn_prox = QPushButton('Próxima')
        self.lbl_nav = QLabel()
        self.lbl_nav.setObjectName('suave')
        self.lbl_nav.setAlignment(Qt.AlignCenter)
        self.btn_ant.clicked.connect(lambda: self._navegar(-1))
        self.btn_prox.clicked.connect(lambda: self._navegar(1))
        ln.addWidget(self.btn_ant)
        ln.addWidget(self.lbl_nav, 1)
        ln.addWidget(self.btn_prox)
        self.nav.hide()
        lp.addWidget(self.nav)

        self.lbl_status = QLabel('Pronto.')
        self.lbl_status.setObjectName('status')
        lp.addWidget(self.lbl_status)

        raiz.addWidget(principal, 1)

        # ---------------- barra lateral
        lateral = QFrame()
        lateral.setObjectName('lateral')
        rolagem = QScrollArea()
        rolagem.setWidget(lateral)
        rolagem.setWidgetResizable(True)
        rolagem.setFixedWidth(300)
        rolagem.setFrameShape(QFrame.NoFrame)
        rolagem.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        raiz.addWidget(rolagem)

        ll = QVBoxLayout(lateral)
        ll.setContentsMargins(14, 14, 14, 14)
        ll.setSpacing(10)

        marca = QLabel('Detector de EPI')
        marca.setObjectName('marca')
        sub = QLabel('Capacete, colete e cabeça')
        sub.setObjectName('suave')
        ll.addWidget(marca)
        ll.addWidget(sub)
        ll.addSpacing(10)

        # fonte
        ll.addWidget(titulo_secao('Fonte'))
        self.cb_fonte = QComboBox()
        for chave, texto in FONTES:
            self.cb_fonte.addItem(texto, chave)
        ll.addWidget(self.cb_fonte)

        self.pilha = QStackedWidget()
        self.sp_camera = QSpinBox()
        self.sp_camera.setRange(0, 9)
        self.pilha.addWidget(self._linha('Câmera nº', self.sp_camera))           # webcam
        self.sp_monitor = QSpinBox()
        self.sp_monitor.setRange(1, 6)
        self.chk_ocultar = QCheckBox('Esconder esta janela da captura')
        self.chk_ocultar.setChecked(True)
        self.chk_ocultar.setToolTip('Evita o efeito espelho quando a janela está no\n'
                                    'monitor capturado (Windows 10 2004 ou superior).')
        self.chk_ocultar.toggled.connect(self._aplicar_ocultar)
        caixa_tela = QWidget()
        lt = QVBoxLayout(caixa_tela)
        lt.setContentsMargins(0, 0, 0, 0)
        lt.addWidget(self._linha('Monitor nº', self.sp_monitor))
        lt.addWidget(self.chk_ocultar)
        if sys.platform != 'win32':
            self.chk_ocultar.hide()
        self.pilha.addWidget(caixa_tela)                                         # tela
        self.ed_video = QLineEdit()
        self.ed_video.setPlaceholderText('Escolha um arquivo .mp4, .avi ...')
        self.pilha.addWidget(self._arquivo(self.ed_video, self._escolher_video))  # video
        self.ed_img = QLineEdit()
        self.ed_img.setPlaceholderText('Imagem ou pasta')
        caixa_img = QWidget()
        li = QVBoxLayout(caixa_img)
        li.setContentsMargins(0, 0, 0, 0)
        li.addWidget(self.ed_img)
        bi = QHBoxLayout()
        b1 = QPushButton('Imagem…')
        b2 = QPushButton('Pasta…')
        b1.clicked.connect(self._escolher_imagem)
        b2.clicked.connect(self._escolher_pasta)
        bi.addWidget(b1)
        bi.addWidget(b2)
        li.addLayout(bi)
        self.pilha.addWidget(caixa_img)                                          # imagens
        self.ed_rtsp = QLineEdit()
        self.ed_rtsp.setPlaceholderText('rtsp://usuario:senha@192.168.0.50/stream')
        self.pilha.addWidget(self.ed_rtsp)                                       # rtsp
        ex = QLabel(f'Pasta: exemplos/')
        ex.setObjectName('suave')
        self.pilha.addWidget(ex)                                                 # exemplos
        ll.addWidget(self.pilha)
        self.cb_fonte.currentIndexChanged.connect(self._trocar_fonte)
        self.cb_fonte.currentIndexChanged.connect(lambda _: self._aplicar_ocultar())
        self._trocar_fonte(0)

        # modelo
        ll.addSpacing(6)
        ll.addWidget(titulo_secao('Modelo'))
        self.cb_modelo = QComboBox()
        pesos = sorted(PASTA_PESOS.glob('*.pt'), key=lambda p: (not p.name.startswith('yolo26'), p.name))
        for p in pesos:
            self.cb_modelo.addItem(p.name, str(p))
        ll.addWidget(self.cb_modelo)

        # confiança
        ll.addSpacing(6)
        topo_conf = QHBoxLayout()
        topo_conf.addWidget(titulo_secao('Confiança mínima'))
        self.lbl_conf = QLabel('0.35')
        self.lbl_conf.setObjectName('conf')
        topo_conf.addStretch()
        topo_conf.addWidget(self.lbl_conf)
        ll.addLayout(topo_conf)
        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setRange(5, 95)
        self.sl_conf.setValue(35)
        self.sl_conf.valueChanged.connect(self._mudar_conf)
        ll.addWidget(self.sl_conf)
        dica = QLabel('Menor = detecta mais (e erra mais). Maior = só o que tem certeza.')
        dica.setObjectName('suave')
        dica.setWordWrap(True)
        ll.addWidget(dica)

        # controles
        ll.addSpacing(10)
        self.btn_iniciar = QPushButton('Iniciar')
        self.btn_iniciar.setObjectName('iniciar')
        self.btn_iniciar.clicked.connect(self._iniciar_parar)
        ll.addWidget(self.btn_iniciar)
        linha_ctrl = QHBoxLayout()
        self.btn_pausa = QPushButton('Pausar')
        self.btn_foto = QPushButton('Salvar quadro')
        self.btn_pausa.clicked.connect(self._pausar)
        self.btn_foto.clicked.connect(self._salvar)
        linha_ctrl.addWidget(self.btn_pausa)
        linha_ctrl.addWidget(self.btn_foto)
        ll.addLayout(linha_ctrl)
        self.chk_gravar = QCheckBox('Gravar vídeo do resultado')
        self.chk_gravar.toggled.connect(self._mudar_gravar)
        ll.addWidget(self.chk_gravar)

        # contadores
        ll.addSpacing(12)
        topo_cont = QHBoxLayout()
        topo_cont.addWidget(titulo_secao('No quadro'))
        topo_cont.addStretch()
        self.lbl_fps = QLabel('')
        self.lbl_fps.setObjectName('suave')
        topo_cont.addWidget(self.lbl_fps)
        ll.addLayout(topo_cont)
        self.contadores = {}
        for chave, rotulo, cor in CLASSES:
            c = Contador(rotulo, cor)
            self.contadores[chave] = c
            ll.addWidget(c)

        ll.addStretch()
        btn_pasta = QPushButton('Abrir pasta de resultados')
        btn_pasta.setObjectName('discreto')
        btn_pasta.clicked.connect(self._abrir_saidas)
        btn_janela = QPushButton('Salvar imagem da janela (F12)')
        btn_janela.setObjectName('discreto')
        btn_janela.clicked.connect(lambda: self._capturar_janela())
        ll.addWidget(btn_janela)
        ll.addWidget(btn_pasta)

        # atalhos globais (os de letra/espaço ficam no keyPressEvent para não
        # atrapalhar a digitação nos campos de texto)
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self._tela_cheia)
        QShortcut(QKeySequence('Ctrl+S'), self, activated=self._salvar)
        QShortcut(QKeySequence(Qt.Key_F12), self, activated=lambda: self._capturar_janela())

        # botões não "roubam" o foco: assim ESPAÇO pausa em vez de clicar no botão
        for b in (self.btn_iniciar, self.btn_pausa, self.btn_foto, self.btn_ant,
                  self.btn_prox, btn_pasta, btn_janela):
            b.setFocusPolicy(Qt.NoFocus)

        self._definir_alerta('neutro', 'Pronto — escolha a fonte e clique em Iniciar')
        self._atualizar_botoes(False)
        if not pesos:
            QMessageBox.warning(self, 'Modelos não encontrados',
                                f'Não encontrei arquivos .pt em\n{PASTA_PESOS}')

    # ------------------------------------------------ montagem
    def _linha(self, texto, widget):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(texto)
        lbl.setObjectName('suave')
        l.addWidget(lbl)
        l.addStretch()
        l.addWidget(widget)
        return w

    def _arquivo(self, edit, acao):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        b = QPushButton('…')
        b.setFixedWidth(40)
        b.clicked.connect(acao)
        l.addWidget(edit, 1)
        l.addWidget(b)
        return w

    def _trocar_fonte(self, i):
        """Mostra só as opções da fonte escolhida, sem reservar espaço para as outras."""
        for k in range(self.pilha.count()):
            pol = QSizePolicy.Preferred if k == i else QSizePolicy.Ignored
            self.pilha.widget(k).setSizePolicy(QSizePolicy.Preferred, pol)
        self.pilha.setCurrentIndex(i)
        self.pilha.setFixedHeight(self.pilha.currentWidget().sizeHint().height())

    # ------------------------------------------------ escolha de arquivos
    def _escolher_video(self):
        filtro = 'Vídeos (' + ' '.join('*' + e for e in sorted(EXT_VID)) + ')'
        arq, _ = QFileDialog.getOpenFileName(self, 'Escolher vídeo', str(Path.home()), filtro)
        if arq:
            self.ed_video.setText(arq)

    def _escolher_imagem(self):
        filtro = 'Imagens (' + ' '.join('*' + e for e in sorted(EXT_IMG)) + ')'
        arq, _ = QFileDialog.getOpenFileName(self, 'Escolher imagem', str(Path.home()), filtro)
        if arq:
            self.ed_img.setText(arq)

    def _escolher_pasta(self):
        pasta = QFileDialog.getExistingDirectory(self, 'Escolher pasta', str(Path.home()))
        if pasta:
            self.ed_img.setText(pasta)

    # ------------------------------------------------ controle
    def _iniciar_parar(self):
        if self.worker is not None and self.worker.isRunning():
            self._parar()
        else:
            self._iniciar()

    def _iniciar(self):
        tipo = self.cb_fonte.currentData()
        valor = None
        if tipo == 'video':
            valor = self.ed_video.text().strip()
            if not valor or not Path(valor).is_file():
                return self._avisar('Escolha um arquivo de vídeo.')
        elif tipo == 'imagens':
            valor = self.ed_img.text().strip()
            if not valor or not Path(valor).exists():
                return self._avisar('Escolha uma imagem ou pasta.')
        elif tipo == 'rtsp':
            valor = self.ed_rtsp.text().strip()
            if not valor:
                return self._avisar('Informe o endereço da câmera.')
        elif tipo == 'exemplos':
            valor = str(PASTA_EXEMPLOS)
        if self.cb_modelo.count() == 0:
            return self._avisar('Nenhum modelo .pt na pasta pesos/.')

        self.galeria, self.indice = [], 0
        self.nav.setVisible(tipo in ('imagens', 'exemplos'))
        self.lbl_nav.setText('')
        self.video.limpar()
        self.video.setText('Carregando o modelo…')
        self._aplicar_ocultar()

        self.worker = Detector(tipo, valor, self.cb_modelo.currentData(),
                               self.sl_conf.value() / 100, self.sp_camera.value(),
                               self.sp_monitor.value(), self.chk_gravar.isChecked())
        self.worker.quadro.connect(self._receber_quadro)
        self.worker.imagem.connect(self._receber_imagem)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.erro.connect(self._erro)
        self.worker.finished.connect(self._terminou)
        self.worker.start()
        self._atualizar_botoes(True)

    def _parar(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.parar()
            self.lbl_status.setText('Parando…')

    def _terminou(self):
        self._atualizar_botoes(False)
        if not self.galeria and self.video._img is None:
            self.video.limpar()
        self.lbl_fps.setText('')

    def _pausar(self):
        if self.worker is None or not self.worker.isRunning():
            return
        pausado = self.worker.alternar_pausa()
        self.btn_pausa.setText('Retomar' if pausado else 'Pausar')
        self.lbl_status.setText('Pausado.' if pausado else 'Rodando.')

    def _salvar(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.salvar_quadro()
        elif self.video._img is not None:
            PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
            destino = PASTA_SAIDA / f'quadro_{datetime.now():%Y%m%d_%H%M%S}.png'
            self.video._img.save(str(destino))
            self.lbl_status.setText(f'Quadro salvo: saidas/{destino.name}')

    def _capturar_janela(self, destino=None):
        """Salva a janela inteira como PNG (o Qt desenha a própria janela,
        então funciona mesmo quando o Print Screen do Windows não pega o app)."""
        if destino is None:
            PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
            destino = PASTA_SAIDA / f'janela_{datetime.now():%Y%m%d_%H%M%S}.png'
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        self.grab().save(str(destino))
        self.lbl_status.setText(f'Janela salva: {Path(destino).relative_to(BASE).as_posix()}')
        return destino

    def _mudar_conf(self, v):
        self.lbl_conf.setText(f'{v / 100:.2f}')
        if self.worker is not None:
            self.worker.conf = v / 100

    def _mudar_gravar(self, ligado):
        if self.worker is not None:
            self.worker.gravar = ligado

    def _atualizar_botoes(self, rodando):
        self.btn_iniciar.setText('Parar' if rodando else 'Iniciar')
        self.btn_iniciar.setProperty('rodando', rodando)
        self.btn_iniciar.style().unpolish(self.btn_iniciar)
        self.btn_iniciar.style().polish(self.btn_iniciar)
        self.btn_pausa.setEnabled(rodando)
        self.btn_pausa.setText('Pausar')
        for w in (self.cb_fonte, self.pilha, self.cb_modelo):
            w.setEnabled(not rodando)

    # ------------------------------------------------ resultados
    def _receber_quadro(self, img, contagem, fps):
        self.video.mostrar(img)
        self._atualizar_contagem(contagem)
        self.lbl_fps.setText(f'{fps:.1f} fps' if fps else '')

    def _receber_imagem(self, img, contagem, nome):
        self.galeria.append((img, contagem, nome))
        if len(self.galeria) == 1:
            self._mostrar_galeria(0)
        else:
            self._atualizar_nav()

    def _mostrar_galeria(self, i):
        if not self.galeria:
            return
        self.indice = max(0, min(i, len(self.galeria) - 1))
        img, contagem, _ = self.galeria[self.indice]
        self.video.mostrar(img)
        self._atualizar_contagem(contagem)
        self._atualizar_nav()

    def _navegar(self, passo):
        if self.nav.isVisible():
            self._mostrar_galeria(self.indice + passo)

    def _atualizar_nav(self):
        total = len(self.galeria)
        nome = self.galeria[self.indice][2] if total else ''
        self.lbl_nav.setText(f'{self.indice + 1} de {total}   ·   {nome}')
        self.btn_ant.setEnabled(self.indice > 0)
        self.btn_prox.setEnabled(self.indice < total - 1)

    def _atualizar_contagem(self, contagem):
        for chave, cartao in self.contadores.items():
            cartao.definir(contagem.get(chave, 0))
        cabecas = contagem.get('head', 0)
        protegidos = contagem.get('helmet', 0)
        if cabecas:
            txt = 'cabeça encontrada' if cabecas == 1 else 'cabeças encontradas'
            self._definir_alerta('perigo', f'Atenção: {cabecas} {txt} no quadro')
        elif protegidos:
            self._definir_alerta('ok', 'Nenhuma cabeça encontrada no quadro')
        else:
            self._definir_alerta('neutro', 'Nenhuma pessoa detectada no quadro')

    def _definir_alerta(self, estado, texto):
        self.alerta.setText(texto)
        self.alerta.setProperty('estado', estado)
        self.alerta.style().unpolish(self.alerta)
        self.alerta.style().polish(self.alerta)

    # ------------------------------------------------ utilidades
    def _erro(self, msg):
        self.lbl_status.setText(msg)
        self._definir_alerta('neutro', 'Não foi possível continuar — veja a mensagem abaixo')
        QMessageBox.warning(self, 'Detector de EPI', msg)

    def _avisar(self, msg):
        self.lbl_status.setText(msg)
        QMessageBox.information(self, 'Detector de EPI', msg)

    def _abrir_saidas(self):
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(PASTA_SAIDA)))

    def _tela_cheia(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _aplicar_ocultar(self):
        """Tira a janela da captura de tela (WDA_EXCLUDEFROMCAPTURE).

        Só vale com a fonte "Captura de tela"; nas outras a janela fica visível
        para o Print Screen / Ferramenta de Captura do Windows.
        """
        if sys.platform != 'win32':
            return
        try:
            import ctypes
            esconder = (self.chk_ocultar.isChecked()
                        and self.cb_fonte.currentData() == 'tela')
            modo = 0x11 if esconder else 0x0
            ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), modo)
        except Exception:
            pass

    def keyPressEvent(self, e):
        tecla = e.key()
        if tecla == Qt.Key_Space:
            self._pausar()
        elif tecla == Qt.Key_S:
            self._salvar()
        elif tecla == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self._parar()
        elif tecla == Qt.Key_Left:
            self._navegar(-1)
        elif tecla == Qt.Key_Right:
            self._navegar(1)
        else:
            super().keyPressEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        self._aplicar_ocultar()

    def closeEvent(self, e):
        if self.worker is not None and self.worker.isRunning():
            self.worker.parar()
            self.worker.wait(3000)
        super().closeEvent(e)


# =================================================================== estilo
ESTILO = """
* { font-family: 'Segoe UI', Arial, sans-serif; font-size: 9pt; color: #1F2328; }
QMainWindow, #principal { background: #FFFFFF; }
#lateral { background: #FFFFFF; border-left: 1px solid #D0D7DE; }
QScrollArea { background: #FFFFFF; }
#marca { font-size: 13pt; font-weight: 600; }
#secao { font-weight: 600; color: #1F2328; margin-top: 4px; }
#suave { color: #656D76; }
#status { color: #656D76; border-top: 1px solid #D0D7DE; padding-top: 6px; }
#conf { font-weight: 600; }

#video { background: #F6F8FA; border: 1px solid #D0D7DE; color: #8C959F; font-size: 10pt; }

#alerta { padding: 8px; font-size: 10pt; font-weight: 600; border: 1px solid #D0D7DE; }
#alerta[estado="neutro"] { background: #F6F8FA; color: #656D76; }
#alerta[estado="ok"]     { background: #DAFBE1; color: #116329; border-color: #9BD8AC; }
#alerta[estado="perigo"] { background: #FFEBE9; color: #A40E26; border-color: #F2A7A7; }

#cartao { background: #FFFFFF; border: 1px solid #D0D7DE; }
#cartaoNome { color: #1F2328; }
#cartaoValor { font-size: 14pt; font-weight: 600; }

QComboBox, QLineEdit, QSpinBox {
    background: #FFFFFF; border: 1px solid #D0D7DE; border-radius: 3px; padding: 4px 6px;
    selection-background-color: #0969DA; selection-color: #FFFFFF;
}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus { border-color: #0969DA; }
QComboBox QAbstractItemView { background: #FFFFFF; border: 1px solid #D0D7DE; }
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled { color: #8C959F; background: #F6F8FA; }

QPushButton {
    background: #F6F8FA; border: 1px solid #D0D7DE; border-radius: 3px; padding: 5px 10px;
}
QPushButton:hover { background: #EEF1F4; }
QPushButton:pressed { background: #E1E5E9; }
QPushButton:disabled { color: #8C959F; }
#iniciar { background: #1F6FEB; border: 1px solid #1A5FCC; color: #FFFFFF; font-weight: 600; padding: 7px; }
#iniciar:hover { background: #1A5FCC; }
#iniciar[rodando="true"] { background: #CF222E; border-color: #A40E26; }
#iniciar[rodando="true"]:hover { background: #A40E26; }

QCheckBox { spacing: 6px; }
QToolTip { background: #FFFFFF; color: #1F2328; border: 1px solid #D0D7DE; padding: 4px; }
"""


def main():
    os.chdir(BASE)  # caminhos relativos (pesos/, saidas/) sempre a partir do projeto
    app = QApplication(sys.argv)
    app.setApplicationName('Detector de EPI')
    app.setStyle('Fusion')
    app.setStyleSheet(ESTILO)
    janela = Janela()
    janela.show()
    if '--print' in sys.argv:
        # demonstração automática: roda nas imagens de exemplo, tira o print e fecha
        janela.cb_fonte.setCurrentIndex(janela.cb_fonte.findData('exemplos'))
        QTimer.singleShot(300, janela._iniciar)

        inicio = time.monotonic()

        def tirar_print():
            # espera a primeira imagem processada (o modelo pode demorar a carregar)
            if not janela.galeria and time.monotonic() - inicio < 180:
                return QTimer.singleShot(500, tirar_print)
            QTimer.singleShot(800, finalizar)  # dá tempo de desenhar contadores/alerta

        def finalizar():
            destino = janela._capturar_janela(PASTA_EXEMPLOS / 'interface.png')
            print(f'Print salvo em {destino}')
            janela.close()
        QTimer.singleShot(1000, tirar_print)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
