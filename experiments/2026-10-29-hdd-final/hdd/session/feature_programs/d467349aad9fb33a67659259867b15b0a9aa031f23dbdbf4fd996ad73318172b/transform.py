def transform(frame):
    raw = frame['smart_197_raw']
    mean_7d = raw.rolling('7D', min_periods=1).mean()
    surge_7d = raw - raw.rolling('7D', min_periods=1).min()
    pos_daily_change = raw.diff(1).clip(lower=0)
    nonzero = (raw > 0).astype('float64').where(raw.notna())
    nonzero_days_7d = nonzero.rolling('7D', min_periods=1).sum()
    return pd.DataFrame({
        'fe_pending_7d_mean': mean_7d,
        'fe_pending_surge_7d': surge_7d,
        'fe_pending_pos_daily_change': pos_daily_change,
        'fe_pending_nonzero_days_7d': nonzero_days_7d
    }, index=frame.index)