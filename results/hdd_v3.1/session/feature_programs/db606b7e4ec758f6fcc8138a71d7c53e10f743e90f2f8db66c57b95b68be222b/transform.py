def transform(frame):
    raw = frame['smart_5_raw']
    norm = frame['smart_5_normalized']
    lifetime_max = raw.expanding(min_periods=1).max()
    recent_rise_7d = raw - raw.rolling('7D', min_periods=1).min()
    positive_daily_jump = raw.diff(1).clip(lower=0)
    norm_decline_7d = norm.rolling('7D', min_periods=1).max() - norm
    return pd.DataFrame({
        'fe_smart5_raw_lifetime_max': lifetime_max,
        'fe_smart5_raw_recent_rise_7d': recent_rise_7d,
        'fe_smart5_raw_positive_daily_jump': positive_daily_jump,
        'fe_smart5_normalized_recent_decline_7d': norm_decline_7d
    }, index=frame.index)