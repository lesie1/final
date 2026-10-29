def transform(frame):
    burst_1h = frame['max_burst_interval_1h']
    burst_15m = frame['max_burst_interval_15m']
    dq_1h = frame['max_dq_interval_1h']
    dq_15m = frame['max_dq_interval_15m']
    baseline_b1h = burst_1h.rolling('7D', min_periods=1).mean()
    baseline_b15m = burst_15m.rolling('7D', min_periods=1).mean()
    baseline_d1h = dq_1h.rolling('7D', min_periods=1).mean()
    baseline_d15m = dq_15m.rolling('7D', min_periods=1).mean()
    fe_burst_interval_decline_1h = np.maximum(baseline_b1h - burst_1h, 0)
    fe_burst_interval_decline_15m = np.maximum(baseline_b15m - burst_15m, 0)
    fe_dq_interval_decline_1h = np.maximum(baseline_d1h - dq_1h, 0)
    fe_dq_interval_decline_15m = np.maximum(baseline_d15m - dq_15m, 0)
    return pd.DataFrame({'fe_burst_interval_decline_1h': fe_burst_interval_decline_1h, 'fe_burst_interval_decline_15m': fe_burst_interval_decline_15m, 'fe_dq_interval_decline_1h': fe_dq_interval_decline_1h, 'fe_dq_interval_decline_15m': fe_dq_interval_decline_15m}, index=frame.index)