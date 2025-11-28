# features.py
import numpy as np
from scipy.signal import hilbert

def rsa_series(r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    RSA(Respiratory Sinus Arrhythmia) 유사 시계열:
    - RR 인터벌(IBI) 편차를 고정 샘플링으로 확장하지 않고
      순수 시퀀스 기반 변동만 반환 (주파수 분석 시 리샘플 필요)
    """
    if r_peaks is None or len(r_peaks) < 3:
        return np.array([])
    ibi = np.diff(r_peaks) / fs  # seconds
    return ibi - np.mean(ibi)

def fm_series(peak_indices: np.ndarray, fs: float) -> np.ndarray:
    """
    FM(Frequency Modulation) 시리즈: 피크 간 간격(IBI/ISI)의 변동.
    ECG는 R-R, PPG는 IPI에 해당.
    """
    if peak_indices is None or len(peak_indices) < 3:
        return np.array([])
    isi = np.diff(peak_indices) / fs
    return isi - np.mean(isi)

def am_series_from_signal(sig: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    """
    AM(Amplitude Modulation) 시리즈: 피크에서의 진폭 궤적
    """
    if sig is None or peak_indices is None or len(peak_indices) < 2:
        return np.array([])
    return sig[peak_indices]

def bm_series_moving_avg(sig: np.ndarray, fs: float, win_sec: float = 2.0) -> np.ndarray:
    """
    BM(Baseline Modulation): 이동평균으로 추정한 저주파 베이스라인
    """
    if sig is None or len(sig) == 0:
        return np.array([])
    n = max(1, int(win_sec * fs))
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(sig, kernel, mode="same")

def envelope_series(sig: np.ndarray) -> np.ndarray:
    """
    힐버트 변환 기반 에너지 엔벨로프(AM 대체/보강용)
    """
    if sig is None or len(sig) == 0:
        return np.array([])
    analytic = hilbert(sig)
    return np.abs(analytic)
