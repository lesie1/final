def transform(frame):
    b15 = frame['error_burst_count_15m']
    d15 = frame['error_dq_count_15m']
    b1h = frame['error_burst_count_1h']
    d1h = frame['error_dq_count_1h']
    p15 = b15 * d15
    p1h = b1h * d1h
    ratio = (p15 + 1.0) / (p1h + 1.0)
    log_p15 = np.log1p(p15)
    return pd.DataFrame({'fe_burst_dq_product_15m': p15, 'fe_burst_dq_product_1h': p1h, 'fe_burst_dq_recent_ratio': ratio, 'fe_burst_dq_log_product_15m': log_p15}, index=frame.index)