# pipeline.py
import numpy as np
from scipy.signal import detrend, resample_poly
from peak import detect_r_peaks, detect_ppg_peaks, amplitude_modulation, baseline_modulation
from features import rsa_series, am_series_from_signal, bm_series_moving_avg, fm_series, envelope_series
from tf_analysis import dominant_freq, bandpower
from fusion import fuse_rr

class RespRatePipeline:
    """
    호흡률 추정 파이프라인 (ECG/PPG 융합)
    - 전처리: detrend/z-score(내부), 피크 검출은 bandpass 포함
    - ECG 기반: RSA(IBI 변동) / FM(RR 변동)
    - PPG 기반: AM(피크 진폭 궤적) / BM(베이스라인 sway) / FM(IPI 변동) / ENV(힐버트 엔벨로프)
    - TF 분석: Welch 기본, 대역 0.1~0.5 Hz
    - SQI: 길이/대역전력/정규성 간단 지표 (0~1)
    - Fusion: SQI 가중 + 불일치 패널티, confidence 반환
    """

    def __init__(self, fs=128, resp_band=(0.1, 0.5), tf_method="welch"):
        self.fs = float(fs)
        self.resp_band = resp_band
        self.tf_method = tf_method
        self._resp_fs = 4.0  # 호흡 시계열 리샘플 목표 FS (분석 안정화용)

    # ---------- helpers ----------
    @staticmethod
    def _zscore(x):
        s = np.std(x)
        return (x - np.mean(x)) / (s if s > 1e-12 else 1.0)

    def _resample_to(self, x: np.ndarray, orig_fs: float, target_fs: float):
        if x is None or len(x) == 0:
            return np.array([]), target_fs
        if abs(orig_fs - target_fs) < 1e-6:
            return x, orig_fs
        # 정수 비율 근사
        up = int(round(target_fs * 100))
        dn = int(round(orig_fs * 100))
        from math import gcd
        g = gcd(up, dn)
        up //= g; dn //= g
        y = resample_poly(x, up, dn)
        return y, target_fs

    def _sqi_len(self, x, min_len=16, max_len=512):
        if x is None:
            return 0.0
        L = len(x)
        return max(0.0, min(1.0, (L - min_len) / max(1, (max_len - min_len))))

    def _sqi_power(self, x, fs):
        bp = bandpower(x, fs, self.resp_band, method=self.tf_method)
        # 전력 → 로지스틱 맵핑(경험적)
        return max(0.0, min(1.0, 1 - np.exp(-bp)))

    def _sqi_flat(self, x):
        if x is None or len(x) < 3:
            return 0.0
        flat = np.mean(np.abs(np.diff(x)) < 1e-6)
        # flat 비율이 낮을수록 좋다
        return max(0.0, min(1.0, 1.0 - flat))

    def _combine_sqi(self, *vals):
        # 보수적 결합: 가중 평균 대신 하모닉 평균 느낌으로 낮은 값을 반영
        vals = [v for v in vals if v is not None]
        if not vals:
            return 0.0
        return float(np.mean(vals) * (1.0 - np.std(vals))) if len(vals) > 1 else float(vals[0])

    # ---------- main ----------
    def process_window(self, ecg: np.ndarray, ppg: np.ndarray):
        if ecg is None or ppg is None or len(ecg) != len(ppg):
            raise ValueError("ECG/PPG 길이가 같아야 합니다.")

        n = len(ecg)
        if n < int(5 * self.fs):
            raise ValueError("분석 윈도우가 너무 짧습니다. 최소 5초 이상 권장.")

        # 1) 전처리(가벼운 detrend만; 피크 검출에서 별도 bandpass)
        ecg_d = detrend(ecg)
        ppg_d = detrend(ppg)

        # 2) 피크 검출
        r_peaks = detect_r_peaks(ecg_d, self.fs)
        p_peaks = detect_ppg_peaks(ppg_d, self.fs)

        # 3) 시리즈 구성 (ECG: RSA/FM, PPG: AM/BM/FM/ENV)
        rsa = rsa_series(r_peaks, self.fs)               # irregular RR
        fm_ecg = fm_series(r_peaks, self.fs)             # RR FM
        am_ppg = amplitude_modulation(ppg_d, p_peaks)    # PPG peak amplitudes
        bm_ppg = baseline_modulation(ppg_d, self.fs)     # moving average baseline
        fm_ppg = fm_series(p_peaks, self.fs)             # IPI FM
        env_ppg = envelope_series(ppg_d)                 # Hilbert envelope

        # 4) 시계열 리샘플(분석 안정화) – RSA/FM/AM/ENV는 불규칙/저주파이므로 낮은 fs로
        target_fs = self._resp_fs
        rsa_rs, _ = self._resample_to(rsa, orig_fs=1.0, target_fs=target_fs) if len(rsa) > 0 else (np.array([]), target_fs)
        fm_ecg_rs, _ = self._resample_to(fm_ecg, orig_fs=1.0, target_fs=target_fs) if len(fm_ecg) > 0 else (np.array([]), target_fs)

        am_ppg_rs, _ = self._resample_to(am_ppg, orig_fs=(len(am_ppg)/ (len(ppg_d)/self.fs)) if len(am_ppg)>1 else target_fs, target_fs=target_fs) if len(am_ppg)>0 else (np.array([]), target_fs)
        env_ppg_rs, _ = self._resample_to(env_ppg, orig_fs=self.fs, target_fs=target_fs) if len(env_ppg)>0 else (np.array([]), target_fs)

        # BM은 이미 원 신호 길이 → 다운샘플
        bm_ppg_rs, _ = self._resample_to(bm_ppg, orig_fs=self.fs, target_fs=target_fs) if len(bm_ppg)>0 else (np.array([]), target_fs)
        fm_ppg_rs, _ = self._resample_to(fm_ppg, orig_fs=1.0, target_fs=target_fs) if len(fm_ppg)>0 else (np.array([]), target_fs)

        # 5) 주파수 분석 (Welch 기본)
        def _hz_to_bpm(f): return f*60 if f is not None else None

        rr_ecg_rsa_hz = dominant_freq(rsa_rs, target_fs, band=self.resp_band, method=self.tf_method) if len(rsa_rs)>0 else None
        rr_ecg_fm_hz  = dominant_freq(fm_ecg_rs, target_fs, band=self.resp_band, method=self.tf_method) if len(fm_ecg_rs)>0 else None

        rr_ppg_am_hz  = dominant_freq(am_ppg_rs, target_fs, band=self.resp_band, method=self.tf_method) if len(am_ppg_rs)>0 else None
        rr_ppg_bm_hz  = dominant_freq(bm_ppg_rs, target_fs, band=self.resp_band, method=self.tf_method) if len(bm_ppg_rs)>0 else None
        rr_ppg_fm_hz  = dominant_freq(fm_ppg_rs, target_fs, band=self.resp_band, method=self.tf_method) if len(fm_ppg_rs)>0 else None
        rr_ppg_env_hz = dominant_freq(env_ppg_rs, target_fs, band=self.resp_band, method=self.tf_method) if len(env_ppg_rs)>0 else None

        rr_ecg_rsa = _hz_to_bpm(rr_ecg_rsa_hz)
        rr_ecg_fm  = _hz_to_bpm(rr_ecg_fm_hz)

        # PPG: 여러 추정치 중 평균(간단) – 이후 SQI로 가중 가능
        ppg_candidates = [x for x in [_hz_to_bpm(rr_ppg_am_hz),
                                      _hz_to_bpm(rr_ppg_bm_hz),
                                      _hz_to_bpm(rr_ppg_fm_hz),
                                      _hz_to_bpm(rr_ppg_env_hz)] if x is not None]
        rr_ppg = float(np.mean(ppg_candidates)) if len(ppg_candidates) else None

        # ECG: RSA 우선, 없으면 FM
        rr_ecg = rr_ecg_rsa if rr_ecg_rsa is not None else rr_ecg_fm

        # 6) SQI 계산
        sqi_ecg = self._combine_sqi(
            self._sqi_len(rsa_rs, 16, 512),
            self._sqi_power(rsa_rs if len(rsa_rs)>0 else fm_ecg_rs, target_fs),
            self._sqi_flat(rsa_rs if len(rsa_rs)>0 else fm_ecg_rs)
        )

        # PPG SQI – 후보들 중 전력/평탄성 평균
        ppg_sigs = [s for s in [am_ppg_rs, bm_ppg_rs, fm_ppg_rs, env_ppg_rs] if len(s)>0]
        if ppg_sigs:
            sqi_parts = []
            for s in ppg_sigs:
                sqi_parts.append(self._sqi_power(s, target_fs))
                sqi_parts.append(self._sqi_flat(s))
            sqi_ppg = float(np.mean(sqi_parts))
        else:
            sqi_ppg = 0.0

        # 7) 융합
        rr_final, confidence = fuse_rr(rr_ecg, rr_ppg, sqi_ecg, sqi_ppg, agree_penalty_bpm=6.0)

        return {
            # 최종
            "RR_final_BPM": rr_final,
            "Confidence_0to1": confidence,

            # ECG 파트
            "RR_ECG_RSA_BPM": rr_ecg_rsa,
            "RR_ECG_FM_BPM": rr_ecg_fm,
            "SQI_ECG": sqi_ecg,
            "N_R_peaks": int(len(r_peaks)),

            # PPG 파트
            "RR_PPG_AM_BPM": _hz_to_bpm(rr_ppg_am_hz),
            "RR_PPG_BM_BPM": _hz_to_bpm(rr_ppg_bm_hz),
            "RR_PPG_FM_BPM": _hz_to_bpm(rr_ppg_fm_hz),
            "RR_PPG_ENV_BPM": _hz_to_bpm(rr_ppg_env_hz),
            "RR_PPG_Fused_BPM": rr_ppg,
            "SQI_PPG": sqi_ppg,
            "N_PPG_peaks": int(len(p_peaks)),

            # 공통 메타
            "Window_sec": n / self.fs,
            "Resp_band_Hz": self.resp_band,
            "TF_method": self.tf_method
        }
