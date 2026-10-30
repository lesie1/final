def transform(frame):
    raw = frame['smart_188_raw']
    missing = raw.notna()
    has_timeout = ((raw > 0) * 1).where(missing)
    recent_nonzero = has_timeout.rolling('7D', min_periods=1).sum().where(missing)
    recent_min = raw.rolling('7D', min_periods=1).min()
    recent_increase = (raw - recent_min).where(missing)
    rolling_mean = raw.rolling('7D', min_periods=1).mean().where(missing)
    expanding_mean = raw.expanding(min_periods=1).mean().where(missing)
    return pd.DataFrame({'fe_188_recent_nonzero_7d_count': recent_nonzero, 'fe_188_recent_increase_7d': recent_increase, 'fe_188_rolling_mean_7d': rolling_mean, 'fe_188_expanding_mean': expanding_mean}, index=frame.index)