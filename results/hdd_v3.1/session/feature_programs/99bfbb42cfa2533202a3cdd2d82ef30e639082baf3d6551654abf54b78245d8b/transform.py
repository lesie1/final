def transform(frame):
    s = frame['smart_196_raw']
    recent_sum = s.rolling('7D', min_periods=1).sum()
    recent_max = s.rolling('7D', min_periods=1).max()
    nonzero_days = (s > 0).astype('float').rolling('7D', min_periods=1).sum()
    daily_delta = s.diff(1)
    return pd.DataFrame({
        'fe_196_recent_sum_7d': recent_sum,
        'fe_196_recent_max_7d': recent_max,
        'fe_196_nonzero_days_7d': nonzero_days,
        'fe_196_daily_delta': daily_delta
    }, index=frame.index)