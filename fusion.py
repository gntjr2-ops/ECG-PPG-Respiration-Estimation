# fusion.py
import math

def _nz(x, default=0.0):
    return x if (x is not None and not (isinstance(x, float) and math.isnan(x))) else default

def fuse_rr(rr_ecg, rr_ppg, sqi_ecg, sqi_ppg, agree_penalty_bpm: float = 6.0):
    """
    가중 융합:
    - 가중치: SQI 기반 (0~1)
    - 일관성 패널티: 두 추정치가 크게 벗어나면(>agree_penalty_bpm) 낮은 SQI 쪽 가중 축소
    - 반환: (rr_final_bpm, confidence 0~1)
    """
    rr_e = _nz(rr_ecg, None)
    rr_p = _nz(rr_ppg, None)

    we = sqi_ecg if rr_e is not None else 0.0
    wp = sqi_ppg if rr_p is not None else 0.0

    if rr_e is None and rr_p is None:
        return 0.0, 0.0

    # 일관성 패널티
    if rr_e is not None and rr_p is not None:
        diff = abs(rr_e - rr_p)
        if diff > agree_penalty_bpm:
            # 차이가 클수록 낮은 SQI를 더 깎음(최소 50% 감쇠)
            if we < wp:
                we *= max(0.5, 1.0 - (diff - agree_penalty_bpm) / (2*agree_penalty_bpm))
            else:
                wp *= max(0.5, 1.0 - (diff - agree_penalty_bpm) / (2*agree_penalty_bpm))

    denom = we + wp
    if denom <= 1e-9:
        # 한쪽만 있을 때
        if rr_e is not None:
            return rr_e, min(1.0, sqi_ecg)
        return rr_p, min(1.0, sqi_ppg)

    rr_fused = (we * rr_e if rr_e is not None else 0.0) + (wp * rr_p if rr_p is not None else 0.0)
    rr_fused /= denom

    # confidence: 총 가중 합(=유효 채널 수와 SQI 반영), 두 추정의 일치성 반영
    base_conf = max(we, wp)
    if rr_e is not None and rr_p is not None:
        diff = abs(rr_e - rr_p)
        match = max(0.0, 1.0 - diff / (2 * agree_penalty_bpm))
        conf = 0.5 * base_conf + 0.5 * match
    else:
        conf = base_conf

    return rr_fused, max(0.0, min(1.0, conf))
