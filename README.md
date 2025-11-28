# ECG–PPG Fusion Respiratory Rate Estimation via Time–Frequency Dynamic Window Analysis

## Overview
This project implements a **real-time respiratory rate (RR) estimation algorithm** that fuses **ECG and PPG signals** from wearable devices.  
The algorithm extracts respiratory components from:

- **ECG R–R interval modulation (RSA: Respiratory Sinus Arrhythmia)**
- **PPG amplitude modulation (AM)**
- **PPG baseline modulation (BM)**

Using a **dynamic time–frequency (TF) analysis window**, the system maintains robust performance even under noise, motion artifacts, and signal loss.

Applications include:
- Sleep monitoring  
- Stress/vagal activity analysis  
- Wearable respiratory tracking  
- Clinical research

---

## Key Features

### 1. Dual-Modal Fusion (ECG + PPG)
- ECG-based RSA extraction  
- PPG-based AM/BM modulation extraction  
- Combined RR estimation superior to single-signal methods  
- Robust to noise and missing segments

---

### 2. Dynamic Window Time-Frequency Analysis
- Window size automatically adjusted based on signal quality (SQI)
- Short window → faster responsiveness  
- Long window → more stability  
- Window range: **20–60 seconds**

---

### 3. Time-Frequency Transform (FFT / STFT / Wavelet)
- RR extracted from dominant peak in **0.1–0.5 Hz band**
- Supports:
  - FFT (fast, lightweight)
  - STFT (time–frequency tracking)
  - Wavelet transform (high noise robustness)
- Designed for embedded/low-power wearable environments

---

### 4. SQI-Based Weighted Fusion
- ECG-SQI & PPG-SQI computed separately  
- Final RR determined by weighted fusion:
  
\[
RR = \frac{w_{ecg} \cdot RR_{ecg} + w_{ppg} \cdot RR_{ppg}}{w_{ecg} + w_{ppg}}
\]

- Low-quality modalities automatically down-weighted

---

### 5. Real-Time & Low-Power Implementation
- Efficient FFT-based approach  
- Suitable for MCU/embedded wearable firmware  
- Supports BLE/UART continuous biosignal streaming

---

## Algorithm Workflow

### 1. ECG-Based RR Candidate Extraction
- R-peak detection  
- R–R interval (RRi) series generation  
- HRV/RSA frequency peak detection  
- RSA in 0.1–0.5 Hz interpreted as respiratory component

---

### 2. PPG-Based RR Candidate Extraction
- Beat-to-beat amplitude modulation (AM) extraction  
- Slow baseline modulation (BM) extraction  
- Frequency analysis of AM/BM to detect respiratory peaks

---

### 3. Dynamic Windowing
- Window size: 20–60 seconds  
- SQI high → short window  
- SQI low → long window  
- Ensures balance of responsiveness and stability  

---

### 4. Time-Frequency Peak Detection
- FFT/STFT of ECG-RSA, PPG-AM, PPG-BM  
- Extract dominant peak in **0.1–0.5 Hz**  
- Output RR candidates:
  - `RR_ecg`
  - `RR_ppg_am`
  - `RR_ppg_bm`

---

### 5. Fusion & Final RR Output
- Compute SQI for ECG & PPG  
- Weighted fusion of candidates  
- Final RR updated every window shift  
- Suitable for real-time monitoring

---

### 6. Data Storage & Analysis
- RR estimates, TF spectra, SQI stored in:
  - `.mat`
  - `.csv`
- Enables:
  - Sleep stage analysis  
  - Stress assessment  
  - Long-term respiratory trend analysis  

---

## Advantages
- **High robustness** via dual-modality fusion  
- **Noise-tolerant** RSA/AM/BM feature extraction  
- **Dynamic windowing** for stable long-term RR tracking  
- **Wearable-ready** low-computational design  
- Applicable to healthcare, sleep research, and biometric monitoring

---

## Example Runtime Pipeline
