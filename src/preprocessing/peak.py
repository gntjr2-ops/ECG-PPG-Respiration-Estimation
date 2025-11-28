# peak.py
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def _bandpass(x: np.ndarray, fs: float, low: float, high: float, order=3):
    ny = 0.5 * fs
    lowc, highc = low / ny, high / ny
    if lowc <= 0: lowc = 1e-6
    if highc >= 1: highc = 0.999999
    b, a = butter(order, [lowc, highc], btype="band")
    return filtfilt(b, a, x)

def _zscore(x: np.ndarray):
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 1e-12 else 1.0)

def detect_r_peaks(ecg: np.ndarray, fs: float):
    """
    견고한 ECG R-peak 검출(간단판):
    - 5~15Hz 밴드패스 + z-score + prominence 기반
    """
    if ecg is None or len(ecg) < int(0.8*fs):
        return np.array([], dtype=int)
    y = _bandpass(ecg, fs, 5.0, 15.0)
    z = _zscore(y)
    peaks, _ = find_peaks(z, distance=int(0.25*fs), prominence=1.0)
    return peaks

def detect_ppg_peaks(ppg: np.ndarray, fs: float):
    """
    견고한 PPG 수축기 피크(AM 타겟):
    - 0.5~5Hz 밴드패스 + z-score + prominence
    """
    if ppg is None or len(ppg) < int(0.8*fs):
        return np.array([], dtype=int)
    y = _bandpass(ppg, fs, 0.5, 5.0)
    z = _zscore(y)
    peaks, _ = find_peaks(z, distance=int(0.3*fs), prominence=0.5)
    return peaks

def amplitude_modulation(ppg: np.ndarray, ppg_peaks: np.ndarray):
    return ppg[ppg_peaks] if ppg is not None and ppg_peaks is not None and len(ppg_peaks) > 0 else np.array([])

def baseline_modulation(ppg: np.ndarray, fs: float, win_sec=2.0):
    if ppg is None or len(ppg) == 0:
        return np.array([])
    n = max(1, int(win_sec * fs))
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(ppg, kernel, mode="same")
