def transform(frame):
    raw = frame['smart_187_raw']
    nonzero = (raw > 0).mask(raw.isna())
    nonzero_7d = nonzero.rolling('7D', min_periods=1).sum()
    positive_change = raw.diff(1).clip(lower=0)
    increase_7d = positive_change.rolling('7D', min_periods=1).sum()
    mean_7d = raw.rolling('7D', min_periods=1).mean()
    expanding_mean = raw.expanding(min_periods=1).mean()
    return pd.DataFrame({
        'fe_187_recent_nonzero_7d_count': nonzero_7d,
        'fe_187_recent_increase_7d': increase_7d,
        'fe_187_rolling_mean_7d': mean_7d,
        'fe_187_expanding_mean': expanding_mean
    }, index=frame.index)