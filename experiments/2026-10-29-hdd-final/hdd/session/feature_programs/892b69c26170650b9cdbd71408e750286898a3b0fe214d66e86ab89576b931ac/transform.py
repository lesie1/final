def transform(frame):
    raw = frame['smart_7_raw']
    rolling_mean = raw.rolling('7D', min_periods=1).mean()
    expanding_mean = raw.expanding(min_periods=1).mean()
    surge_ratio = rolling_mean / (expanding_mean + 1)
    log_raw = np.log1p(raw)
    return pd.DataFrame({'fe_s7_rolling_mean_7d': rolling_mean, 'fe_s7_expanding_mean': expanding_mean, 'fe_s7_surge_ratio': surge_ratio, 'fe_s7_log1p_raw': log_raw}, index=frame.index)