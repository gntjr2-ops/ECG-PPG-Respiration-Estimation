# viewer_pg.py
import sys
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
import pyqtgraph as pg
from scipy.signal import welch

def _ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app

def show_resp_dashboard(t, ecg, ppg, fs, res=None, resp_band=(0.1, 0.5)):
    """
    PyQtGraph 대시보드:
      - 상단: ECG/PPG 파형(시간영역, X축 링크, 십자선)
      - 하단: ECG/PPG PSD(Welch), 호흡대역 강조(두 플롯간 동기화)
      - 하단 컨트롤바(QWidget): Autoscale 버튼
    """
    app = _ensure_app()
    pg.setConfigOptions(antialias=True)

    # ===== 메인 QWidget 레이아웃 =====
    win = QWidget()
    win.setWindowTitle("Respiratory Rate Dashboard")
    win.resize(1200, 800)
    vbox = QVBoxLayout(win)

    # 그래픽 씬 위젯
    glw = pg.GraphicsLayoutWidget()
    vbox.addWidget(glw)

    # ----------- 상단: 시간영역 -----------
    p_ecg = glw.addPlot(title="ECG (time)")
    p_ppg = glw.addPlot(title="PPG (time)")
    p_ppg.setXLink(p_ecg)

    for p in (p_ecg, p_ppg):
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setDownsampling(mode="peak")
        p.setClipToView(True)
        p.setLabel('bottom', 'Time (s)')
        p.setLabel('left', 'Amplitude')

    p_ecg.plot(t, ecg, pen=pg.mkPen('#44aaff'))
    p_ppg.plot(t, ppg, pen=pg.mkPen('#55dd88'))

    # 마우스 십자선(동기 이동)
    vline1 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888'))
    vline2 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888'))
    p_ecg.addItem(vline1, ignoreBounds=True)
    p_ppg.addItem(vline2, ignoreBounds=True)

    def _mouse_move(pos):
        if p_ecg.sceneBoundingRect().contains(pos):
            x = p_ecg.vb.mapSceneToView(pos).x()
            vline1.setPos(x); vline2.setPos(x)
    p_ecg.scene().sigMouseMoved.connect(_mouse_move)
    p_ppg.scene().sigMouseMoved.connect(_mouse_move)

    # ----------- 하단: 주파수영역 -----------
    glw.nextRow()
    p_ecg_psd = glw.addPlot(title="ECG PSD (Welch)")
    p_ppg_psd = glw.addPlot(title="PPG PSD (Welch)")
    p_ppg_psd.setXLink(p_ecg_psd)

    for p in (p_ecg_psd, p_ppg_psd):
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setLabel('bottom', 'Frequency (Hz)')
        p.setLabel('left', 'Power (dB)')

    # Welch PSD 유틸: nperseg/overlap 안전 설정
    def _welch(x, fs):
        n = len(x)
        desired = min(2048, max(256, n // 2))
        nperseg = min(n, desired)
        noverlap = min(nperseg // 2, nperseg - 1)
        f, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        return f, psd

    f1, Pxx1 = _welch(ecg, fs)
    f2, Pxx2 = _welch(ppg, fs)
    fmax = 2.0
    m1 = f1 <= fmax
    m2 = f2 <= fmax
    p_ecg_psd.plot(f1[m1], 10*np.log10(Pxx1[m1] + 1e-18), pen=pg.mkPen('#44aaff'), name="ECG")
    p_ppg_psd.plot(f2[m2], 10*np.log10(Pxx2[m2] + 1e-18), pen=pg.mkPen('#55dd88'), name="PPG")

    # 호흡 대역 강조: 두 플롯에 '각각' region 생성 + 동기화
    region_lock = {"busy": False}

    band_region_ecg = pg.LinearRegionItem(values=resp_band, movable=False, brush=(100,100,255,30))
    band_region_ppg = pg.LinearRegionItem(values=resp_band, movable=False, brush=(100,100,255,30))
    p_ecg_psd.addItem(band_region_ecg)
    p_ppg_psd.addItem(band_region_ppg)

    def _sync_from_ecg():
        if region_lock["busy"]: return
        region_lock["busy"] = True
        try:
            band_region_ppg.setRegion(band_region_ecg.getRegion())
        finally:
            region_lock["busy"] = False

    def _sync_from_ppg():
        if region_lock["busy"]: return
        region_lock["busy"] = True
        try:
            band_region_ecg.setRegion(band_region_ppg.getRegion())
        finally:
            region_lock["busy"] = False

    band_region_ecg.sigRegionChanged.connect(_sync_from_ecg)
    band_region_ppg.sigRegionChanged.connect(_sync_from_ppg)

    # 추정 호흡률 수직선
    def _add_rr_line(plot, bpm, name, color):
        if bpm is None: return None
        hz = bpm / 60.0
        line = pg.InfiniteLine(pos=hz, angle=90, pen=pg.mkPen(color, width=2))
        plot.addItem(line)
        # 간단 범례용: 라벨 표시
        txt = pg.TextItem(html=f'<span style="color:{color}">{name}: {bpm:.2f} BPM</span>', anchor=(0,1))
        plot.addItem(txt)
        txt.setPos(hz, plot.viewRange()[1][1])  # 상단 근처
        return line

    if res:
        _add_rr_line(p_ecg_psd, res.get("RR_final_BPM"),     "Final", "#ffaa00")
        _add_rr_line(p_ecg_psd, res.get("RR_ECG_RSA_BPM"),   "ECG_RSA", "#ff66aa")
        _add_rr_line(p_ppg_psd, res.get("RR_PPG_Fused_BPM"), "PPG_Fused", "#33cccc")

    # ===== 하단 컨트롤 바(QWidget) =====
    hbox = QHBoxLayout()
    vbox.addLayout(hbox)
    btn_autoscale = QPushButton("Autoscale All")
    hbox.addWidget(btn_autoscale)
    hbox.addStretch(1)

    def _autoscale():
        for p in (p_ecg, p_ppg, p_ecg_psd, p_ppg_psd):
            p.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
    btn_autoscale.clicked.connect(_autoscale)

    win.show()
    app.exec()
