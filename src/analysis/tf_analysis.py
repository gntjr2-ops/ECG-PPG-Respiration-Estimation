# tf_analysis.py
import numpy as np
from scipy.signal import welch

def _valid_vec(x):
    return x is not None and isinstance(x, (list, tuple, np.ndarray)) and len(x) > 1

def dominant_freq(signal, fs, band=(0.1, 0.5), method="welch"):
    """
    호흡 대역(기본 0.1~0.5Hz)에서의 지배 주파수(Hz) 추정
    - method: "welch" | "fft"
    - 길이가 짧으면 None 반환
    """
    if not _valid_vec(signal) or fs <= 0:
        return None

    x = np.asarray(signal, dtype=float)
    n = len(x)

    if method == "welch":
        # nperseg은 호흡주기 최소 2~3개 확보되도록 설정(경험치)
        nperseg = max(128, min(1024, (n // 4) * 2))
        freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2, detrend="constant")
    else:
        win = np.hanning(n)
        spec = np.abs(np.fft.rfft(x * win))
        freqs = np.fft.rfftfreq(n, 1/fs)
        psd = spec

    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return None
    idx = np.argmax(psd[mask])
    f_dom = freqs[mask][idx]
    return float(f_dom)

def bandpower(signal, fs, band=(0.1, 0.5), method="welch"):
    if not _valid_vec(signal) or fs <= 0:
        return 0.0
    x = np.asarray(signal, dtype=float)
    if method == "welch":
        freqs, psd = welch(x, fs=fs, nperseg=min(2048, max(256, len(x)//2)))
    else:
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))**2
        freqs = np.fft.rfftfreq(len(x), 1/fs)
        psd = spec
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    # 적분 근사
    df = np.mean(np.diff(freqs[mask])) if np.sum(mask) > 1 else (freqs[mask][0] / max(1, len(freqs)))
    return float(np.sum(psd[mask]) * df)
