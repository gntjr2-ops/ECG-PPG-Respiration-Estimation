# demo.py
import numpy as np
from pipeline import RespRatePipeline
from viewer_pg import show_resp_dashboard   # ← 추가

fs = 128
t = np.arange(0, 60, 1/fs)

# 예시 호흡률(0.25 Hz → 15 BPM)
rr_hz = 0.25
resp = np.sin(2*np.pi*rr_hz*t)

# 합성 ECG/PPG (호흡 성분 AM/BM/FM 혼합)
rng = np.random.default_rng(42)
ecg_carrier = np.sin(2*np.pi*1.3*t)          # ECG 캐리어(심장 주기)
ppg_carrier = np.sin(2*np.pi*1.3*t + 0.5)    # PPG 캐리어
ecg = ecg_carrier*(1 + 0.10*resp) + 0.04*rng.standard_normal(len(t))
ppg = (ppg_carrier*(1 + 0.20*resp) +
       0.10*np.sin(2*np.pi*rr_hz*t + 0.3) +  # baseline sway
       0.05*rng.standard_normal(len(t)))

pipe = RespRatePipeline(fs=fs)
res = pipe.process_window(ecg, ppg)

print("=== 호흡률 분석 결과 ===")
for k, v in res.items():
    print(f"{k}: {v}")

show_resp_dashboard(t, ecg, ppg, fs, res=res, resp_band=(0.1, 0.5))